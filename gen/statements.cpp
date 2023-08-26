//===-- statements.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/hdrgen.h"
#include "dmd/id.h"
#include "dmd/identifier.h"
#include "dmd/import.h"
#include "dmd/init.h"
#include "dmd/mangle.h"
#include "dmd/module.h"
#include "dmd/mtype.h"
#include "dmd/root/port.h"
#include "gen/abi/abi.h"
#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/coverage.h"
#include "gen/dcompute/target.h"
#include "gen/dvalue.h"
#include "gen/funcgenstate.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/recursivevisitor.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include "ir/irmodule.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InlineAsm.h"
#include <fstream>
#include <math.h>
#include <stdio.h>

//////////////////////////////////////////////////////////////////////////////
// FIXME: Integrate these functions
void GccAsmStatement_toIR(GccAsmStatement *stmt, IRState *irs);
void AsmStatement_toIR(InlineAsmStatement *stmt, IRState *irs);
void CompoundAsmStatement_toIR(CompoundAsmStatement *stmt, IRState *p);

//////////////////////////////////////////////////////////////////////////////

namespace {
bool isAssertFalse(Expression *e) {
  return e ? e->type == Type::tnoreturn &&
                 (e->op == EXP::halt || e->op == EXP::assert_)
           : false;
}

bool isAssertFalse(Statement *s) {
  if (!s)
    return false;
  if (auto es = s->isExpStatement())
    return isAssertFalse(es->exp);
  else if (auto ss = s->isScopeStatement())
    return isAssertFalse(ss->statement);
  return false;
}
}

//////////////////////////////////////////////////////////////////////////////

/// Used to check if a control-flow stmt body contains any label. A label
/// is considered anything that lets us jump inside the body _apart from_
/// the stmt. That includes case / default statements.
/// It is a StoppableVisitor that stops when a label is found.
/// It's to be passed in a ContainsLabelWalker which recursively
/// walks the tree and updates our `inside_switch` flag accordingly.
struct ContainsLabelVisitor : public StoppableVisitor {
  // If RecursiveWalker finds a SwitchStatement,
  // `insideSwitch` points to that statement.
  SwitchStatement *insideSwitch = nullptr;

  using StoppableVisitor::visit;

  void visit(Statement *stmt) override {}

  void visit(LabelStatement *stmt) override { stop = true; }

  void visit(CaseStatement *stmt) override {
    if (insideSwitch == nullptr)
      stop = true;
  }

  void visit(DefaultStatement *stmt) override {
    if (insideSwitch == nullptr)
      stop = true;
  }

  bool foundLabel() { return stop; }

  void visit(Declaration *) override {}
  void visit(Initializer *) override {}
  void visit(Dsymbol *) override {}
  void visit(Expression *) override {}
};

/// As the RecursiveWalker, but it gets a ContainsLabelVisitor
/// and updates its `insideSwitch` field accordingly.
class ContainsLabelWalker : public RecursiveWalker {
public:
  using RecursiveWalker::visit;

  explicit ContainsLabelWalker(ContainsLabelVisitor *visitor,
                               bool _continueAfterStop = true)
      : RecursiveWalker(visitor, _continueAfterStop) {}

  void visit(SwitchStatement *stmt) override {
    ContainsLabelVisitor *ev = static_cast<ContainsLabelVisitor *>(v);
    SwitchStatement *save = ev->insideSwitch;
    ev->insideSwitch = stmt;
    RecursiveWalker::visit(stmt);
    ev->insideSwitch = save;
  }

  void visit(Expression *) override {}
};

class ToIRVisitor : public Visitor {
  IRState *irs;

public:
  explicit ToIRVisitor(IRState *irs) : irs(irs) {}

  //////////////////////////////////////////////////////////////////////////

  // Import all functions from class Visitor
  using Visitor::visit;

  //////////////////////////////////////////////////////////////////////////

  void visit(CompoundStatement *stmt) override {
    IF_LOG Logger::println("CompoundStatement::toIR(): %s",
                           stmt->loc.toChars());
    LOG_SCOPE;

    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);

    for (auto s : *stmt->statements) {
      if (s) {
        s->accept(this);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(ReturnStatement *stmt) override {
    IF_LOG Logger::println("ReturnStatement::toIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;

    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);

    // emit dwarf stop point
    irs->DBuilder.EmitStopPoint(stmt->loc);

    emitCoverageLinecountInc(stmt->loc);

    // The LLVM value to return, or null for void returns.
    LLValue *returnValue = nullptr;

    auto &funcGen = irs->funcGen();
    IrFunction *const f = &funcGen.irFunc;
    FuncDeclaration *const fd = f->decl;
    llvm::FunctionType *funcType = f->getLLVMFuncType();

    emitInstrumentationFnLeave(fd);

    const auto cleanupScopeBeforeExpression =
        funcGen.scopes.currentCleanupScope();

    // is there a return value expression?
    const bool isMainFunc = isAnyMainFunction(fd);
    if (stmt->exp || isMainFunc) {
      // We clean up manually (*not* using toElemDtor) as the expression might
      // be an lvalue pointing into a temporary, and we may need a load. So we
      // need to make sure to destruct any temporaries after all of that.

      const auto rt = f->type->next;
      const auto rtb = rt->toBasetype();

      if (!stmt->exp) {
        // implicitly return 0 for the main function
        returnValue = LLConstant::getNullValue(funcType->getReturnType());
      } else if ((rtb->ty == TY::Tvoid || rtb->ty == TY::Tnoreturn) &&
                 !isMainFunc) {
        // evaluate expression for side effects
        assert(stmt->exp->type->toBasetype()->ty == TY::Tvoid ||
               stmt->exp->type->toBasetype()->ty == TY::Tnoreturn);
        toElem(stmt->exp);
      } else if (funcType->getReturnType()->isVoidTy()) {
        // if the IR function's return type is void (but not the D one), it uses
        // sret
        assert(!f->type->isref());

        LLValue *sretPointer = f->sretArg;
        assert(sretPointer);

        assert(!f->irFty.arg_sret->rewrite &&
               "ABI shouldn't have to rewrite sret returns");
        DLValue returnValue(rt, sretPointer);

        // try to construct the return value in-place
        const bool constructed = toInPlaceConstruction(&returnValue, stmt->exp);
        if (!constructed) {
          DValue *e = toElem(stmt->exp);

          // store the return value unless NRVO already used the sret pointer
          if (!e->isLVal() || DtoLVal(e) != sretPointer) {
            // call postblit if the expression is a D lvalue
            // exceptions: NRVO and special __result variable (out contracts)
            bool doPostblit = !(fd->isNRVO() && fd->nrvo_var);
            if (doPostblit) {
              if (auto ve = stmt->exp->isVarExp())
                if (ve->var->isResult())
                  doPostblit = false;
            }

            DtoAssign(stmt->loc, &returnValue, e, EXP::blit);
            if (doPostblit)
              callPostblit(stmt->loc, stmt->exp, sretPointer);
          }
        }
      } else {
        // the return type is not void, so this is a normal "register" return
        if (stmt->exp->op == EXP::null_) {
          stmt->exp->type = rt;
        }
        DValue *dval = nullptr;
        // call postblit if necessary
        if (!f->type->isref()) {
          dval = toElem(stmt->exp);
          LLValue *vthis =
              (DtoIsInMemoryOnly(dval->type) ? DtoLVal(dval) : DtoRVal(dval));
          callPostblit(stmt->loc, stmt->exp, vthis);
        } else {
          Expression *ae = stmt->exp;
          dval = toElem(ae);
        }
        // do abi specific transformations on the return value
        returnValue = getIrFunc(fd)->irFty.putRet(dval);

        // Hack around LDC assuming structs and static arrays are in memory:
        // If the function returns a struct or a static array, and the return
        // value is a pointer to a struct or a static array, load from it
        // before returning.
        if (returnValue->getType() != funcType->getReturnType() &&
            DtoIsInMemoryOnly(rt) && isaPointer(returnValue)) {
          Logger::println("Loading value for return");
          returnValue = DtoLoad(funcType->getReturnType(), returnValue);
        }

        // can happen for classes
        if (returnValue->getType() != funcType->getReturnType()) {
          returnValue =
              irs->ir->CreateBitCast(returnValue, funcType->getReturnType());
          IF_LOG Logger::cout()
              << "return value after cast: " << *returnValue << '\n';
        }
      }
    } else {
      // no return value expression means it's a void function.
      assert(funcType->getReturnType()->isVoidTy());
    }

    // If there are no cleanups to run, we try to keep the IR simple and
    // just directly emit the return instruction. If there are cleanups to run
    // first, we need to store the return value to a stack slot, in which case
    // we can use a shared return bb for all these cases.
    const bool useRetValSlot = funcGen.scopes.currentCleanupScope() != 0;
    const bool sharedRetBlockExists = !!funcGen.retBlock;
    if (useRetValSlot) {
      if (!sharedRetBlockExists) {
        funcGen.retBlock = irs->insertBB("return");
        if (returnValue) {
          funcGen.retValSlot =
              DtoRawAlloca(returnValue->getType(), 0, "return.slot");
        }
      }

      // Create the store to the slot at the end of our current basic
      // block, before we run the cleanups.
      if (returnValue) {
        irs->ir->CreateStore(returnValue, funcGen.retValSlot);
      }

      // Now run the cleanups.
      funcGen.scopes.runCleanups(0, funcGen.retBlock);
      // Pop the cleanups pushed during evaluation of the return expression.
      funcGen.scopes.popCleanups(cleanupScopeBeforeExpression);

      irs->ir->SetInsertPoint(funcGen.retBlock);
    }

    // If we need to emit the actual return instruction, do so.
    if (!useRetValSlot || !sharedRetBlockExists) {
      if (returnValue) {
        // Hack: the frontend generates 'return 0;' as last statement of
        // 'void main()'. But the debug location is missing. Use the end
        // of function as debug location.
        if (isAnyMainFunction(fd) && !stmt->loc.linnum()) {
          irs->DBuilder.EmitStopPoint(fd->endloc);
        }

        irs->ir->CreateRet(
            useRetValSlot ? DtoLoad(funcType->getReturnType(), funcGen.retValSlot)
                          : returnValue);
      } else {
        irs->ir->CreateRetVoid();
      }
    }

    // Finally, create a new predecessor-less dummy bb as the current IRScope
    // to make sure we do not emit any extra instructions after the terminating
    // instruction (ret or branch to return bb), which would be illegal IR.
    irs->ir->SetInsertPoint(irs->insertBB("dummy.afterreturn"));
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(ExpStatement *stmt) override {
    IF_LOG Logger::println("ExpStatement::toIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;

    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);

    // emit dwarf stop point
    irs->DBuilder.EmitStopPoint(stmt->loc);

    if (auto e = stmt->exp) {
      if (e->hasCode() &&
          !isAssertFalse(e)) { // `assert(0)` not meant to be covered
        emitCoverageLinecountInc(stmt->loc);
      }

      DValue *elem;
      // a cast(void) around the expression is allowed, but doesn't require any
      // code
      if (e->op == EXP::cast_ && e->type == Type::tvoid) {
        elem = toElemDtor(static_cast<CastExp *>(e)->e1);
      } else {
        elem = toElemDtor(e);
      }
      delete elem;
    }
  }

  //////////////////////////////////////////////////////////////////////////

  bool dcomputeReflectMatches(CallExp *ce) {
    auto arg1 = (DComputeTarget::ID)(*ce->arguments)[0]->toInteger();
    auto arg2 = (*ce->arguments)[1]->toInteger();
    auto dct = irs->dcomputetarget;
    if (!dct) {
      return arg1 == DComputeTarget::ID::Host;
    } else {
      return arg1 == dct->target &&
             (!arg2 || arg2 == static_cast<dinteger_t>(dct->tversion));
    }
  }

  //////////////////////////////////////////////////////////////////////////

  bool containsLabel(Statement *stmt) {
    if (!stmt)
      return false;
    ContainsLabelVisitor labelChecker;
    ContainsLabelWalker walker(&labelChecker, false);
    stmt->accept(&walker);
    return labelChecker.foundLabel();
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(IfStatement *stmt) override {
    IF_LOG Logger::println("IfStatement::toIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;

    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);
    auto truecount = PGO.getRegionCount(stmt);
    auto elsecount = PGO.getCurrentRegionCount() - truecount;
    auto brweights = PGO.createProfileWeights(truecount, elsecount);

    // start a dwarf lexical block
    irs->DBuilder.EmitBlockStart(stmt->loc);
    emitCoverageLinecountInc(stmt->loc);
    // Open a new scope for the optional condition variable (`if (auto i = ...)`)
    irs->funcGen().localVariableLifetimeAnnotator.pushScope();


    // This is a (dirty) hack to get codegen time conditional
    // compilation, on account of the fact that we are trying
    // to target multiple backends "simultaneously" with one
    // pass through the front end, to have a single "static"
    // context.
    if (auto ce = stmt->condition->isCallExp()) {
      if (ce->f && ce->f->ident == Id::dcReflect) {
        if (dcomputeReflectMatches(ce))
          stmt->ifbody->accept(this);
        else if (stmt->elsebody)
          stmt->elsebody->accept(this);
        return;
      }
    }
    DValue *cond_e = toElemDtor(stmt->condition);
    LLValue *cond_val = DtoRVal(cond_e);
    // Is it constant?
    if (LLConstant *const_val = llvm::dyn_cast<LLConstant>(cond_val)) {
      Statement *executed = stmt->ifbody;
      Statement *skipped = stmt->elsebody;
      if (const_val->isZeroValue()) {
        std::swap(executed, skipped);
      }
      if (!containsLabel(skipped)) {
        IF_LOG Logger::println("Constant true/false condition - elide.");
        if (executed) {
          irs->DBuilder.EmitBlockStart(executed->loc);
        }
        // True condition, the branch is taken so emit counter increment.
        if (!const_val->isZeroValue()) {
          PGO.emitCounterIncrement(stmt);
        }
        if (executed) {
          executed->accept(this);
          irs->DBuilder.EmitBlockEnd();
        }
        // end the dwarf lexical block
        irs->DBuilder.EmitBlockEnd();
        return;
      }
    }
    llvm::BasicBlock *ifbb = irs->insertBB("if");
    llvm::BasicBlock *endbb = irs->insertBBAfter(ifbb, "endif");
    llvm::BasicBlock *elsebb =
        stmt->elsebody ? irs->insertBBAfter(ifbb, "else") : endbb;

    if (!cond_val->getType()->isIntegerTy(1)) {
      IF_LOG Logger::cout() << "if conditional: " << *cond_val << '\n';
      cond_val = DtoRVal(DtoCast(stmt->loc, cond_e, Type::tbool));
    }
    auto brinstr =
        llvm::BranchInst::Create(ifbb, elsebb, cond_val, irs->scopebb());
    PGO.addBranchWeights(brinstr, brweights);

    // replace current scope
    irs->ir->SetInsertPoint(ifbb);

    // do scoped statements

    if (stmt->ifbody) {
      irs->DBuilder.EmitBlockStart(stmt->ifbody->loc);
      PGO.emitCounterIncrement(stmt);
      stmt->ifbody->accept(this);
      irs->DBuilder.EmitBlockEnd();
    }
    if (!irs->scopereturned()) {
      llvm::BranchInst::Create(endbb, irs->scopebb());
    }

    if (stmt->elsebody) {
      irs->ir->SetInsertPoint(elsebb);
      irs->DBuilder.EmitBlockStart(stmt->elsebody->loc);
      stmt->elsebody->accept(this);
      if (!irs->scopereturned()) {
        llvm::BranchInst::Create(endbb, irs->scopebb());
      }
      irs->DBuilder.EmitBlockEnd();
    }

    // end the dwarf lexical block
    irs->DBuilder.EmitBlockEnd();

    // rewrite the scope
    irs->ir->SetInsertPoint(endbb);
    // Close the scope for the optional condition variable. This is suboptimal,
    // because the condition variable is not in scope in the else block.
    irs->funcGen().localVariableLifetimeAnnotator.popScope();
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(ScopeStatement *stmt) override {
    IF_LOG Logger::println("ScopeStatement::toIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;

    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);

    if (stmt->statement) {
      irs->funcGen().localVariableLifetimeAnnotator.pushScope();
      irs->DBuilder.EmitBlockStart(stmt->statement->loc);
      stmt->statement->accept(this);
      irs->DBuilder.EmitBlockEnd();
      irs->funcGen().localVariableLifetimeAnnotator.popScope();
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(WhileStatement *stmt) override {
    IF_LOG Logger::println("WhileStatement::toIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;

    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);

    // start a dwarf lexical block
    irs->DBuilder.EmitBlockStart(stmt->loc);

    // create while blocks

    llvm::BasicBlock *whilebb = irs->insertBB("whilecond");
    llvm::BasicBlock *whilebodybb = irs->insertBBAfter(whilebb, "whilebody");
    llvm::BasicBlock *endbb = irs->insertBBAfter(whilebodybb, "endwhile");

    // move into the while block
    irs->ir->CreateBr(whilebb);

    // replace current scope
    irs->ir->SetInsertPoint(whilebb);

    // create the condition
    emitCoverageLinecountInc(stmt->condition->loc);
    DValue *cond_e = toElemDtor(stmt->condition);
    LLValue *cond_val = DtoRVal(DtoCast(stmt->loc, cond_e, Type::tbool));
    delete cond_e;

    // conditional branch
    auto branchinst =
        llvm::BranchInst::Create(whilebodybb, endbb, cond_val, irs->scopebb());
    {
      auto loopcount = PGO.getRegionCount(stmt);
      auto brweights =
          PGO.createProfileWeightsWhileLoop(stmt->condition, loopcount);
      PGO.addBranchWeights(branchinst, brweights);
    }

    // rewrite scope
    irs->ir->SetInsertPoint(whilebodybb);

    // while body code
    irs->funcGen().jumpTargets.pushLoopTarget(stmt, whilebb, endbb);
    PGO.emitCounterIncrement(stmt);
    if (stmt->_body) {
      stmt->_body->accept(this);
    }
    irs->funcGen().jumpTargets.popLoopTarget();

    // loop
    if (!irs->scopereturned()) {
      llvm::BranchInst::Create(whilebb, irs->scopebb());
    }

    // rewrite the scope
    irs->ir->SetInsertPoint(endbb);

    // end the dwarf lexical block
    irs->DBuilder.EmitBlockEnd();
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(DoStatement *stmt) override {
    IF_LOG Logger::println("DoStatement::toIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;

    auto &PGO = irs->funcGen().pgo;
    auto entryCount = PGO.setCurrentStmt(stmt);

    // start a dwarf lexical block
    irs->DBuilder.EmitBlockStart(stmt->loc);

    // create while blocks
    llvm::BasicBlock *dowhilebb = irs->insertBB("dowhile");
    llvm::BasicBlock *condbb = irs->insertBBAfter(dowhilebb, "dowhilecond");
    llvm::BasicBlock *endbb = irs->insertBBAfter(condbb, "enddowhile");

    // move into the while block
    assert(!irs->scopereturned());
    llvm::BranchInst::Create(dowhilebb, irs->scopebb());

    // replace current scope
    irs->ir->SetInsertPoint(dowhilebb);

    // do-while body code
    irs->funcGen().jumpTargets.pushLoopTarget(stmt, condbb, endbb);
    PGO.emitCounterIncrement(stmt);
    if (stmt->_body) {
      stmt->_body->accept(this);
    }
    irs->funcGen().jumpTargets.popLoopTarget();

    // branch to condition block
    llvm::BranchInst::Create(condbb, irs->scopebb());
    irs->ir->SetInsertPoint(condbb);

    // create the condition
    emitCoverageLinecountInc(stmt->condition->loc);
    DValue *cond_e = toElemDtor(stmt->condition);
    LLValue *cond_val = DtoRVal(DtoCast(stmt->loc, cond_e, Type::tbool));
    delete cond_e;

    // conditional branch
    auto branchinst =
        llvm::BranchInst::Create(dowhilebb, endbb, cond_val, irs->scopebb());
    {
      // The region counter includes fallthrough from the previous statement.
      // Subtract parent count to get the true branch count of the loop
      // conditional.
      auto loopcount = PGO.getRegionCount(stmt) - entryCount;
      auto brweights =
          PGO.createProfileWeightsWhileLoop(stmt->condition, loopcount);
      PGO.addBranchWeights(branchinst, brweights);
    }

    // rewrite the scope
    irs->ir->SetInsertPoint(endbb);

    // end the dwarf lexical block
    irs->DBuilder.EmitBlockEnd();
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(ForStatement *stmt) override {
    IF_LOG Logger::println("ForStatement::toIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;

    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);

    // start new dwarf lexical block
    irs->DBuilder.EmitBlockStart(stmt->loc);
    irs->funcGen().localVariableLifetimeAnnotator.pushScope();

    // create for blocks
    llvm::BasicBlock *forbb = irs->insertBB("forcond");
    llvm::BasicBlock *forbodybb = irs->insertBBAfter(forbb, "forbody");
    llvm::BasicBlock *forincbb = irs->insertBBAfter(forbodybb, "forinc");
    llvm::BasicBlock *endbb = irs->insertBBAfter(forincbb, "endfor");

    // init
    if (stmt->_init != nullptr) {
      stmt->_init->accept(this);
    }

    // move into the for condition block, ie. start the loop
    assert(!irs->scopereturned());
    llvm::BranchInst::Create(forbb, irs->scopebb());

    // In case of loops that have been rewritten to a composite statement
    // containing the initializers and then the actual loop, we need to
    // register the former as target scope start.
    Statement *scopeStart = stmt->getRelatedLabeled();
    while (ScopeStatement *scope = scopeStart->isScopeStatement()) {
      scopeStart = scope->statement;
    }
    irs->funcGen().jumpTargets.pushLoopTarget(scopeStart, forincbb, endbb);

    // replace current scope
    irs->ir->SetInsertPoint(forbb);

    // create the condition
    llvm::Value *cond_val;
    if (stmt->condition) {
      emitCoverageLinecountInc(stmt->condition->loc);
      DValue *cond_e = toElemDtor(stmt->condition);
      cond_val = DtoRVal(DtoCast(stmt->loc, cond_e, Type::tbool));
      delete cond_e;
    } else {
      cond_val = DtoConstBool(true);
    }

    // conditional branch
    assert(!irs->scopereturned());
    auto branchinst =
        llvm::BranchInst::Create(forbodybb, endbb, cond_val, irs->scopebb());
    {
      auto brweights = PGO.createProfileWeightsForLoop(stmt);
      PGO.addBranchWeights(branchinst, brweights);
    }

    // rewrite scope
    irs->ir->SetInsertPoint(forbodybb);

    // do for body code
    PGO.emitCounterIncrement(stmt);
    if (stmt->_body) {
      stmt->_body->accept(this);
    }

    // move into the for increment block
    if (!irs->scopereturned()) {
      llvm::BranchInst::Create(forincbb, irs->scopebb());
    }
    irs->ir->SetInsertPoint(forincbb);

    // increment
    if (stmt->increment) {
      emitCoverageLinecountInc(stmt->increment->loc);
      DValue *inc = toElemDtor(stmt->increment);
      delete inc;
    }

    // loop
    if (!irs->scopereturned()) {
      llvm::BranchInst::Create(forbb, irs->scopebb());
    }

    irs->funcGen().jumpTargets.popLoopTarget();

    // rewrite the scope
    irs->ir->SetInsertPoint(endbb);

    // end the dwarf lexical block
    irs->funcGen().localVariableLifetimeAnnotator.popScope();
    irs->DBuilder.EmitBlockEnd();
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(BreakStatement *stmt) override {
    IF_LOG Logger::println("BreakStatement::toIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;

    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);

    // don't emit two terminators in a row
    // happens just before DMD generated default statements if the last case
    // terminates
    if (irs->scopereturned()) {
      return;
    }

    // emit dwarf stop point
    irs->DBuilder.EmitStopPoint(stmt->loc);

    emitCoverageLinecountInc(stmt->loc);

    if (stmt->ident) {
      IF_LOG Logger::println("ident = %s", stmt->ident->toChars());

      // Get the loop or break statement the label refers to
      Statement *targetStatement = stmt->target->statement;
      ScopeStatement *tmp;
      while ((tmp = targetStatement->isScopeStatement())) {
        targetStatement = tmp->statement;
      }

      irs->funcGen().jumpTargets.breakToStatement(targetStatement);
    } else {
      irs->funcGen().jumpTargets.breakToClosest();
    }

    // the break terminated this basicblock, start a new one
    llvm::BasicBlock *bb = irs->insertBB("afterbreak");
    irs->ir->SetInsertPoint(bb);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(ContinueStatement *stmt) override {
    IF_LOG Logger::println("ContinueStatement::toIR(): %s",
                           stmt->loc.toChars());
    LOG_SCOPE;

    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);

    // emit dwarf stop point
    irs->DBuilder.EmitStopPoint(stmt->loc);

    emitCoverageLinecountInc(stmt->loc);

    if (stmt->ident) {
      IF_LOG Logger::println("ident = %s", stmt->ident->toChars());

      // get the loop statement the label refers to
      Statement *targetLoopStatement = stmt->target->statement;
      ScopeStatement *tmp;
      while ((tmp = targetLoopStatement->isScopeStatement())) {
        targetLoopStatement = tmp->statement;
      }

      irs->funcGen().jumpTargets.continueWithLoop(targetLoopStatement);
    } else {
      irs->funcGen().jumpTargets.continueWithClosest();
    }

    // the continue terminated this basicblock, start a new one
    llvm::BasicBlock *bb = irs->insertBB("aftercontinue");
    irs->ir->SetInsertPoint(bb);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(ScopeGuardStatement *stmt) override {
    stmt->error("Internal Compiler Error: ScopeGuardStatement should have been "
                "lowered by frontend.");
    fatal();
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(TryFinallyStatement *stmt) override {
    IF_LOG Logger::println("TryFinallyStatement::toIR(): %s",
                           stmt->loc.toChars());
    LOG_SCOPE;

    auto &PGO = irs->funcGen().pgo;
    /*auto entryCount = */ PGO.setCurrentStmt(stmt);

    // emit dwarf stop point
    irs->DBuilder.EmitStopPoint(stmt->loc);

    // We only need to consider exception handling/cleanup issues if there
    // is both a try and a finally block. If not, just directly emit what
    // is present.
    if (!stmt->_body || !stmt->finalbody) {
      if (stmt->_body) {
        irs->DBuilder.EmitBlockStart(stmt->_body->loc);
        stmt->_body->accept(this);
        irs->DBuilder.EmitBlockEnd();
      } else if (stmt->finalbody) {
        irs->DBuilder.EmitBlockStart(stmt->finalbody->loc);
        stmt->finalbody->accept(this);
        irs->DBuilder.EmitBlockEnd();
      }
      return;
    }

    // We'll append the "try" part to the current basic block later. No need
    // for an extra one (we'd need to branch to it unconditionally anyway).
    llvm::BasicBlock *trybb = irs->scopebb();

    llvm::BasicBlock *finallybb = irs->insertBB("finally");
    // Create a block to branch to after successfully running the try block
    // and any cleanups.
    llvm::BasicBlock *successbb =
        irs->scopereturned() ? nullptr
                             : irs->insertBBAfter(finallybb, "try.success");

    // Emit the finally block and set up the cleanup scope for it.
    irs->ir->SetInsertPoint(finallybb);
    irs->DBuilder.EmitBlockStart(stmt->finalbody->loc);
    stmt->finalbody->accept(this);
    irs->DBuilder.EmitBlockEnd();
    CleanupCursor cleanupBefore;

    // For @compute code, don't emit any exception handling as there are no
    // exceptions anyway.
    const bool computeCode = !!irs->dcomputetarget;
    if (!computeCode) {
      cleanupBefore = irs->funcGen().scopes.currentCleanupScope();
      irs->funcGen().scopes.pushCleanup(finallybb, irs->scopebb());
    }
    // Emit the try block.
    irs->ir->SetInsertPoint(trybb);

    assert(stmt->_body);
    irs->DBuilder.EmitBlockStart(stmt->_body->loc);
    stmt->_body->accept(this);
    irs->DBuilder.EmitBlockEnd();

    if (successbb) {
      if (!computeCode)
        irs->funcGen().scopes.runCleanups(cleanupBefore, successbb);
      irs->ir->SetInsertPoint(successbb);
      // PGO counter tracks the continuation of the try-finally statement
      PGO.emitCounterIncrement(stmt);
    }
    if (!computeCode)
      irs->funcGen().scopes.popCleanups(cleanupBefore);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(TryCatchStatement *stmt) override {
    IF_LOG Logger::println("TryCatchStatement::toIR(): %s",
                           stmt->loc.toChars());
    LOG_SCOPE;
    assert(!irs->dcomputetarget);

    auto &PGO = irs->funcGen().pgo;

    // Emit dwarf stop point
    irs->DBuilder.EmitStopPoint(stmt->loc);

    // We'll append the "try" part to the current basic block later. No need
    // for an extra one (we'd need to branch to it unconditionally anyway).
    llvm::BasicBlock *trybb = irs->scopebb();

    // Create a basic block to branch to after leaving the try or an
    // associated catch block successfully.
    llvm::BasicBlock *endbb = irs->insertBB("try.success.or.caught");

    irs->funcGen().scopes.pushTryCatch(stmt, endbb);

    // Emit the try block.
    irs->ir->SetInsertPoint(trybb);

    assert(stmt->_body);
    irs->DBuilder.EmitBlockStart(stmt->_body->loc);
    stmt->_body->accept(this);
    irs->DBuilder.EmitBlockEnd();

    if (!irs->scopereturned())
      llvm::BranchInst::Create(endbb, irs->scopebb());

    irs->funcGen().scopes.popTryCatch();

    irs->ir->SetInsertPoint(endbb);

    // PGO counter tracks the continuation of the try statement
    PGO.emitCounterIncrement(stmt);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(ThrowStatement *stmt) override {
    IF_LOG Logger::println("ThrowStatement::toIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;
    assert(!irs->dcomputetarget);

    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);

    // emit dwarf stop point
    irs->DBuilder.EmitStopPoint(stmt->loc);

    emitCoverageLinecountInc(stmt->loc);

    assert(stmt->exp);
    DtoThrow(stmt->loc, toElemDtor(stmt->exp));
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(SwitchStatement *stmt) override {
    IF_LOG Logger::println("SwitchStatement::toIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;

    auto &funcGen = irs->funcGen();

    auto &PGO = funcGen.pgo;
    PGO.setCurrentStmt(stmt);
    const auto incomingPGORegionCount = PGO.getCurrentRegionCount();

    irs->DBuilder.EmitStopPoint(stmt->loc);
    emitCoverageLinecountInc(stmt->loc);
    llvm::BasicBlock *const oldbb = irs->scopebb();

    // The cases of the switch statement, in codegen order.
    auto cases = stmt->cases;
    const auto caseCount = cases->length;

    // llvm::Values for the case indices. Might not be llvm::Constants for
    // runtime-initialised immutable globals as case indices, in which case we
    // need to emit a `br` chain instead of `switch`.
    llvm::SmallVector<llvm::Value *, 16> indices;
    indices.reserve(caseCount);
    bool useSwitchInst = true;

    for (auto cs : *cases) {
      auto ce = cs->exp;
      if (auto ceConst = tryToConstElem(ce, irs)) {
        indices.push_back(ceConst);
      } else {
        indices.push_back(DtoRVal(toElemDtor(ce)));
        useSwitchInst = false;
      }
    }
    assert(indices.size() == caseCount);

    // body block.
    // FIXME: that block is never used
    llvm::BasicBlock *bodybb = irs->insertBB("switchbody");

    // end (break point)
    llvm::BasicBlock *endbb = irs->insertBBAfter(bodybb, "switchend");

    // default
    auto defaultTargetBB = endbb;
    if (stmt->sdefault) {
      Logger::println("has default");
      defaultTargetBB =
          funcGen.switchTargets.getOrCreate(stmt->sdefault, "default", *irs);
    }

    // do switch body
    assert(stmt->_body);
    irs->ir->SetInsertPoint(bodybb);
    funcGen.jumpTargets.pushBreakTarget(stmt, endbb);
    stmt->_body->accept(this);
    funcGen.jumpTargets.popBreakTarget();
    if (!irs->scopereturned()) {
      llvm::BranchInst::Create(endbb, irs->scopebb());
    }

    irs->ir->SetInsertPoint(oldbb);
    if (useSwitchInst) {
      // The case index value.
      LLValue *condVal = DtoRVal(toElemDtor(stmt->condition));

      // Create switch and add the cases.
      // For PGO instrumentation, we need to add counters /before/ the case
      // statement bodies, because the counters should only count the jumps
      // directly from the switch statement and not "goto default", etc.
      llvm::SwitchInst *si;
      if (!PGO.emitsInstrumentation()) {
        si = llvm::SwitchInst::Create(condVal, defaultTargetBB, caseCount,
                                      irs->scopebb());
        for (size_t i = 0; i < caseCount; ++i) {
          si->addCase(isaConstantInt(indices[i]),
                      funcGen.switchTargets.get((*cases)[i]));
        }
      } else {
        auto switchbb = irs->scopebb();
        // Add PGO instrumentation.
        // Create "default" counter bb.
        {
          llvm::BasicBlock *defaultcntr =
              irs->insertBBBefore(defaultTargetBB, "defaultcntr");
          irs->ir->SetInsertPoint(defaultcntr);
          if (stmt->sdefault)
              PGO.emitCounterIncrement(stmt->sdefault);
          llvm::BranchInst::Create(defaultTargetBB, defaultcntr);
          // Create switch
          si = llvm::SwitchInst::Create(condVal, defaultcntr, caseCount,
                                        switchbb);
        }

        // Create and add case counter bbs.
        for (size_t i = 0; i < caseCount; ++i) {
          const auto cs = (*cases)[i];
          const auto body = funcGen.switchTargets.get(cs);

          auto casecntr = irs->insertBBBefore(body, "casecntr");
          irs->ir->SetInsertPoint(casecntr);
          PGO.emitCounterIncrement(cs);
          llvm::BranchInst::Create(body, casecntr);
          si->addCase(isaConstantInt(indices[i]), casecntr);
        }
      }

      // Apply PGO switch branch weights:
      {
        // Get case statements execution counts from profile data.
        std::vector<uint64_t> case_prof_counts;
        case_prof_counts.push_back(
            stmt->sdefault ? PGO.getRegionCount(stmt->sdefault) : 0);
        for (auto cs : *cases) {
          auto w = PGO.getRegionCount(cs);
          case_prof_counts.push_back(w);
        }

        auto brweights = PGO.createProfileWeights(case_prof_counts);
        PGO.addBranchWeights(si, brweights);
      }
    } else {
      // We can't use switch, so we will use a bunch of br instructions
      // instead.

      DValue *cond = toElemDtor(stmt->condition);
      LLValue *condVal = DtoRVal(cond);

      llvm::BasicBlock *nextbb = irs->insertBBBefore(endbb, "checkcase");
      llvm::BranchInst::Create(nextbb, irs->scopebb());

      if (stmt->sdefault && PGO.emitsInstrumentation()) {
        // Prepend extra BB to "default:" to increment profiling counter.
        llvm::BasicBlock *defaultcntr =
            irs->insertBBBefore(defaultTargetBB, "defaultcntr");
        irs->ir->SetInsertPoint(defaultcntr);
        PGO.emitCounterIncrement(stmt->sdefault);
        llvm::BranchInst::Create(defaultTargetBB, defaultcntr);
        defaultTargetBB = defaultcntr;
      }

      irs->ir->SetInsertPoint(nextbb);
      auto failedCompareCount = incomingPGORegionCount;
      for (size_t i = 0; i < caseCount; ++i) {
        LLValue *cmp = irs->ir->CreateICmp(llvm::ICmpInst::ICMP_EQ, indices[i],
                                           condVal, "checkcase");
        nextbb = irs->insertBBBefore(endbb, "checkcase");

        // Add case counters for PGO in front of case body
        const auto cs = (*cases)[i];
        auto casejumptargetbb = funcGen.switchTargets.get(cs);
        if (PGO.emitsInstrumentation()) {
          llvm::BasicBlock *casecntr =
              irs->insertBBBefore(casejumptargetbb, "casecntr");
          const auto savedInsertPoint = irs->saveInsertPoint();
          irs->ir->SetInsertPoint(casecntr);
          PGO.emitCounterIncrement(cs);
          llvm::BranchInst::Create(casejumptargetbb, casecntr);
          casejumptargetbb = casecntr;
        }

        // Create the comparison branch for this case
        auto branchinst = llvm::BranchInst::Create(casejumptargetbb, nextbb,
                                                   cmp, irs->scopebb());

        // Calculate and apply PGO branch weights
        {
          auto trueCount = PGO.getRegionCount(cs);
          assert(trueCount <= failedCompareCount &&
                 "Higher branch count than switch incoming count!");
          failedCompareCount -= trueCount;
          auto brweights =
              PGO.createProfileWeights(trueCount, failedCompareCount);
          PGO.addBranchWeights(branchinst, brweights);
        }

        irs->ir->SetInsertPoint(nextbb);
      }

      llvm::BranchInst::Create(defaultTargetBB, irs->scopebb());
    }

    irs->ir->SetInsertPoint(endbb);
    // PGO counter tracks exit point of switch statement:
    PGO.emitCounterIncrement(stmt);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(CaseStatement *stmt) override {
    IF_LOG Logger::println("CaseStatement::toIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;

    auto &funcGen = irs->funcGen();
    auto &PGO = funcGen.pgo;
    PGO.setCurrentStmt(stmt);

    const auto body = funcGen.switchTargets.getOrCreate(stmt, "case", *irs);
    // The BB may have already been created by a `goto case` statement.
    // Move it after the current scope BB for lexical order.
    body->moveAfter(irs->scopebb());

    if (!irs->scopereturned()) {
      llvm::BranchInst::Create(body, irs->scopebb());
    }

    irs->ir->SetInsertPoint(body);

    assert(stmt->statement);
    irs->DBuilder.EmitBlockStart(stmt->statement->loc);
    if (!isAssertFalse(stmt->statement)) {
      emitCoverageLinecountInc(stmt->loc);
    }
    if (stmt->gototarget) {
      PGO.emitCounterIncrement(PGO.getCounterPtr(stmt, 1));
    }
    stmt->statement->accept(this);
    irs->DBuilder.EmitBlockEnd();
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(DefaultStatement *stmt) override {
    IF_LOG Logger::println("DefaultStatement::toIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;

    auto &funcGen = irs->funcGen();
    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);

    const auto body = funcGen.switchTargets.getOrCreate(stmt, "default", *irs);
    // The BB may have already been created.
    // Move it after the current scope BB for lexical order.
    body->moveAfter(irs->scopebb());

    if (!irs->scopereturned()) {
      llvm::BranchInst::Create(body, irs->scopebb());
    }

    irs->ir->SetInsertPoint(body);

    assert(stmt->statement);
    irs->DBuilder.EmitBlockStart(stmt->statement->loc);
    if (!isAssertFalse(stmt->statement)) {
      emitCoverageLinecountInc(stmt->loc);
    }
    if (stmt->gototarget) {
      PGO.emitCounterIncrement(PGO.getCounterPtr(stmt, 1));
    }
    stmt->statement->accept(this);
    irs->DBuilder.EmitBlockEnd();
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(UnrolledLoopStatement *stmt) override {
    IF_LOG Logger::println("UnrolledLoopStatement::toIR(): %s",
                           stmt->loc.toChars());
    LOG_SCOPE;

    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);

    // if no statements, there's nothing to do
    if (!stmt->statements || !stmt->statements->length) {
      return;
    }

    // start a dwarf lexical block
    irs->DBuilder.EmitBlockStart(stmt->loc);

    // DMD doesn't fold stuff like continue/break, and since this isn't really a
    // loop we have to keep track of each statement and jump to the next/end
    // on continue/break

    // create end block
    llvm::BasicBlock *endbb = irs->insertBB("unrolledend");

    // create a block for each statement
    size_t nstmt = stmt->statements->length;
    llvm::SmallVector<llvm::BasicBlock *, 4> blocks(nstmt, nullptr);
    for (size_t i = 0; i < nstmt; i++)
      blocks[i] = irs->insertBBBefore(endbb, "unrolledstmt");

    // enter first stmt
    if (!irs->scopereturned()) {
      irs->ir->CreateBr(blocks[0]);
    }

    // do statements
    Statement **stmts = &(*stmt->statements)[0];

    for (size_t i = 0; i < nstmt; i++) {
      Statement *s = stmts[i];

      // get blocks
      llvm::BasicBlock *thisbb = blocks[i];
      llvm::BasicBlock *nextbb = (i + 1 == nstmt) ? endbb : blocks[i + 1];

      // update scope
      irs->ir->SetInsertPoint(thisbb);

      // push loop scope
      // continue goes to next statement, break goes to end
      irs->funcGen().jumpTargets.pushLoopTarget(stmt, nextbb, endbb);

      PGO.emitCounterIncrement(s);

      // do statement
      s->accept(this);

      // pop loop scope
      irs->funcGen().jumpTargets.popLoopTarget();

      // next stmt
      if (!irs->scopereturned()) {
        irs->ir->CreateBr(nextbb);
      }
    }

    irs->ir->SetInsertPoint(endbb);

    // PGO counter tracks the continuation after the loop
    PGO.emitCounterIncrement(stmt);

    // end the dwarf lexical block
    irs->DBuilder.EmitBlockEnd();
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(ForeachStatement *stmt) override {
    IF_LOG Logger::println("ForeachStatement::toIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;

    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);

    // start a dwarf lexical block
    irs->DBuilder.EmitBlockStart(stmt->loc);

    // assert(arguments->length == 1);
    assert(stmt->value != 0);
    assert(stmt->aggr != 0);
    assert(stmt->func != 0);

    // Argument* arg = static_cast<Argument*>(arguments->data[0]);
    // Logger::println("Argument is %s", arg->toChars());

    IF_LOG Logger::println("aggr = %s", stmt->aggr->toChars());

    // key
    LLType *keytype = stmt->key ? DtoType(stmt->key->type) : DtoSize_t();
    LLValue *keyvar;
    if (stmt->key) {
      keyvar = DtoRawVarDeclaration(stmt->key);
    } else {
      keyvar = DtoRawAlloca(keytype, 0, "foreachkey");
    }
    LLValue *zerokey = LLConstantInt::get(keytype, 0, false);

    // value
    IF_LOG Logger::println("value = %s", stmt->value->toPrettyChars());
    LLValue *valvar = nullptr;
    if (!stmt->value->isRef() && !stmt->value->isOut()) {
      // Create a local variable to serve as the value.
      DtoRawVarDeclaration(stmt->value);
      valvar = getIrLocal(stmt->value)->value;
    }

    // what to iterate
    DValue *aggrval = toElemDtor(stmt->aggr);

    // get length and pointer
    LLValue *niters = DtoArrayLen(aggrval);
    LLValue *val = DtoArrayPtr(aggrval);

    if (niters->getType() != keytype) {
      size_t sz1 = getTypeBitSize(niters->getType());
      size_t sz2 = getTypeBitSize(keytype);
      if (sz1 < sz2) {
        niters = irs->ir->CreateZExt(niters, keytype, "foreachtrunckey");
      } else if (sz1 > sz2) {
        niters = irs->ir->CreateTrunc(niters, keytype, "foreachtrunckey");
      } else {
        niters = irs->ir->CreateBitCast(niters, keytype, "foreachtrunckey");
      }
    }

    if (stmt->op == TOK::foreach_) {
      new llvm::StoreInst(zerokey, keyvar, irs->scopebb());
    } else {
      new llvm::StoreInst(niters, keyvar, irs->scopebb());
    }

    llvm::BasicBlock *condbb = irs->insertBB("foreachcond");
    llvm::BasicBlock *bodybb = irs->insertBBAfter(condbb, "foreachbody");
    llvm::BasicBlock *nextbb = irs->insertBBAfter(bodybb, "foreachnext");
    llvm::BasicBlock *endbb = irs->insertBBAfter(nextbb, "foreachend");

    llvm::BranchInst::Create(condbb, irs->scopebb());

    // condition
    irs->ir->SetInsertPoint(condbb);

    LLValue *done = nullptr;
    LLValue *load = DtoLoad(keytype, keyvar);
    if (stmt->op == TOK::foreach_) {
      done = irs->ir->CreateICmpULT(load, niters);
    } else if (stmt->op == TOK::foreach_reverse_) {
      done = irs->ir->CreateICmpUGT(load, zerokey);
      load = irs->ir->CreateSub(load, LLConstantInt::get(keytype, 1, false));
      DtoStore(load, keyvar);
    }
    auto branchinst =
        llvm::BranchInst::Create(bodybb, endbb, done, irs->scopebb());
    {
      auto brweights = PGO.createProfileWeightsForeach(stmt);
      PGO.addBranchWeights(branchinst, brweights);
    }

    // init body
    irs->ir->SetInsertPoint(bodybb);
    PGO.emitCounterIncrement(stmt);

    // get value for this iteration
    LLValue *loadedKey = DtoLoad(keytype, keyvar);
    LLValue *gep = DtoGEP1(DtoMemType(aggrval->type->nextOf()), val, loadedKey);

    if (!stmt->value->isRef() && !stmt->value->isOut()) {
      // Copy value to local variable, and use it as the value variable.
      DLValue dst(stmt->value->type, valvar);
      DLValue src(stmt->value->type, gep);
      DtoAssign(stmt->loc, &dst, &src, EXP::assign);
      getIrLocal(stmt->value)->value = valvar;
    } else {
      // Use the GEP as the address of the value variable.
      DtoRawVarDeclaration(stmt->value, gep);
    }

    // emit body
    irs->funcGen().jumpTargets.pushLoopTarget(stmt, nextbb, endbb);
    if (stmt->_body) {
      stmt->_body->accept(this);
    }
    irs->funcGen().jumpTargets.popLoopTarget();

    if (!irs->scopereturned()) {
      llvm::BranchInst::Create(nextbb, irs->scopebb());
    }

    // next
    irs->ir->SetInsertPoint(nextbb);
    if (stmt->op == TOK::foreach_) {
      LLValue *load = DtoLoad(keytype, keyvar);
      load = irs->ir->CreateAdd(load, LLConstantInt::get(keytype, 1, false));
      DtoStore(load, keyvar);
    }
    llvm::BranchInst::Create(condbb, irs->scopebb());

    // end the dwarf lexical block
    irs->DBuilder.EmitBlockEnd();

    // end
    irs->ir->SetInsertPoint(endbb);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(ForeachRangeStatement *stmt) override {
    IF_LOG Logger::println("ForeachRangeStatement::toIR(): %s",
                           stmt->loc.toChars());
    LOG_SCOPE;

    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);

    // start a dwarf lexical block
    irs->DBuilder.EmitBlockStart(stmt->loc);

    // evaluate lwr/upr
    assert(stmt->lwr->type->isintegral());
    LLValue *lower = DtoRVal(toElemDtor(stmt->lwr));
    assert(stmt->upr->type->isintegral());
    LLValue *upper = DtoRVal(toElemDtor(stmt->upr));

    // handle key
    assert(stmt->key->type->isintegral());
    LLValue *keyval  = DtoRawVarDeclaration(stmt->key);
    LLType  *keytype = DtoType(stmt->key->type);
    // store initial value in key
    if (stmt->op == TOK::foreach_) {
      DtoStore(lower, keyval);
    } else {
      DtoStore(upper, keyval);
    }

    // set up the block we'll need
    llvm::BasicBlock *condbb = irs->insertBB("foreachrange_cond");
    llvm::BasicBlock *bodybb = irs->insertBBAfter(condbb, "foreachrange_body");
    llvm::BasicBlock *nextbb = irs->insertBBAfter(bodybb, "foreachrange_next");
    llvm::BasicBlock *endbb = irs->insertBBAfter(nextbb, "foreachrange_end");

    // jump to condition
    llvm::BranchInst::Create(condbb, irs->scopebb());

    // CONDITION
    irs->ir->SetInsertPoint(condbb);

    // first we test that lwr < upr
    lower = DtoLoad(keytype, keyval);
    assert(lower->getType() == upper->getType());
    llvm::ICmpInst::Predicate cmpop;
    if (isLLVMUnsigned(stmt->key->type)) {
      cmpop = (stmt->op == TOK::foreach_) ? llvm::ICmpInst::ICMP_ULT
                                          : llvm::ICmpInst::ICMP_UGT;
    } else {
      cmpop = (stmt->op == TOK::foreach_) ? llvm::ICmpInst::ICMP_SLT
                                          : llvm::ICmpInst::ICMP_SGT;
    }
    LLValue *cond = irs->ir->CreateICmp(cmpop, lower, upper);

    // jump to the body if range is ok, to the end if not
    auto branchinst =
        llvm::BranchInst::Create(bodybb, endbb, cond, irs->scopebb());
    {
      auto brweights = PGO.createProfileWeightsForeachRange(stmt);
      PGO.addBranchWeights(branchinst, brweights);
    }

    // BODY
    irs->ir->SetInsertPoint(bodybb);
    PGO.emitCounterIncrement(stmt);

    // reverse foreach decrements here
    if (stmt->op == TOK::foreach_reverse_) {
      LLValue *v = DtoLoad(keytype, keyval);
      LLValue *one = LLConstantInt::get(v->getType(), 1, false);
      v = irs->ir->CreateSub(v, one);
      DtoStore(v, keyval);
    }

    // emit body
    irs->funcGen().jumpTargets.pushLoopTarget(stmt, nextbb, endbb);
    if (stmt->_body) {
      stmt->_body->accept(this);
    }
    irs->funcGen().jumpTargets.popLoopTarget();

    // jump to next iteration
    if (!irs->scopereturned()) {
      llvm::BranchInst::Create(nextbb, irs->scopebb());
    }

    // NEXT
    irs->ir->SetInsertPoint(nextbb);

    // forward foreach increments here
    if (stmt->op == TOK::foreach_) {
      LLValue *v = DtoLoad(keytype, keyval);
      LLValue *one = LLConstantInt::get(v->getType(), 1, false);
      v = irs->ir->CreateAdd(v, one);
      DtoStore(v, keyval);
    }

    // jump to condition
    llvm::BranchInst::Create(condbb, irs->scopebb());

    // end the dwarf lexical block
    irs->DBuilder.EmitBlockEnd();

    // END
    irs->ir->SetInsertPoint(endbb);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(LabelStatement *stmt) override {
    IF_LOG Logger::println("LabelStatement::toIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;

    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);

    // if it's an inline asm label, we don't create a basicblock, just emit it
    // in the asm
    if (irs->asmBlock) {
      auto a = new IRAsmStmt;
      std::stringstream label;
      printLabelName(label, mangleExact(irs->func()->decl),
                     stmt->ident->toChars());
      label << ":";
      a->code = label.str();
      irs->asmBlock->s.push_back(a);
      irs->asmBlock->internalLabels.push_back(stmt->ident);

      // disable inlining
      irs->func()->setNeverInline();
    } else {
      llvm::BasicBlock *labelBB =
          irs->insertBB(llvm::Twine("label.") + stmt->ident->toChars());
      irs->funcGen().jumpTargets.addLabelTarget(stmt->ident, labelBB);

      if (!irs->scopereturned()) {
        llvm::BranchInst::Create(labelBB, irs->scopebb());
      }

      irs->ir->SetInsertPoint(labelBB);
    }

    PGO.emitCounterIncrement(stmt);
    // statement == nullptr when the label is at the end of function
    if (stmt->statement) {
      stmt->statement->accept(this);
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(GotoStatement *stmt) override {
    IF_LOG Logger::println("GotoStatement::toIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;

    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);

    irs->DBuilder.EmitStopPoint(stmt->loc);

    emitCoverageLinecountInc(stmt->loc);

    DtoGoto(stmt->loc, stmt->label);

    // TODO: Should not be needed.
    llvm::BasicBlock *bb = irs->insertBB("aftergoto");
    irs->ir->SetInsertPoint(bb);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(GotoDefaultStatement *stmt) override {
    IF_LOG Logger::println("GotoDefaultStatement::toIR(): %s",
                           stmt->loc.toChars());
    LOG_SCOPE;

    auto &funcGen = irs->funcGen();
    auto &PGO = funcGen.pgo;
    PGO.setCurrentStmt(stmt);

    irs->DBuilder.EmitStopPoint(stmt->loc);

    emitCoverageLinecountInc(stmt->loc);

    assert(!irs->scopereturned());

    const auto defaultBB = funcGen.switchTargets.get(stmt->sw->sdefault);
    llvm::BranchInst::Create(defaultBB, irs->scopebb());

    // TODO: Should not be needed.
    llvm::BasicBlock *bb = irs->insertBB("aftergotodefault");
    irs->ir->SetInsertPoint(bb);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(GotoCaseStatement *stmt) override {
    IF_LOG Logger::println("GotoCaseStatement::toIR(): %s",
                           stmt->loc.toChars());
    LOG_SCOPE;

    auto &funcGen = irs->funcGen();
    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);

    irs->DBuilder.EmitStopPoint(stmt->loc);

    emitCoverageLinecountInc(stmt->loc);

    assert(!irs->scopereturned());

    const auto caseBB =
        funcGen.switchTargets.getOrCreate(stmt->cs, "goto_case", *irs);
    llvm::BranchInst::Create(caseBB, irs->scopebb());

    // TODO: Should not be needed.
    llvm::BasicBlock *bb = irs->insertBB("aftergotocase");
    irs->ir->SetInsertPoint(bb);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(WithStatement *stmt) override {
    IF_LOG Logger::println("WithStatement::toIR(): %s", stmt->loc.toChars());
    LOG_SCOPE;

    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);

    irs->DBuilder.EmitBlockStart(stmt->loc);

    assert(stmt->exp);

    // with(..) can either be used with expressions or with symbols
    // wthis == null indicates the symbol form
    if (stmt->wthis) {
      LLValue *mem = DtoRawVarDeclaration(stmt->wthis);
      DValue *e = toElemDtor(stmt->exp);
      LLValue *val = (DtoIsInMemoryOnly(e->type) ? DtoLVal(e) : DtoRVal(e));
      DtoStore(val, mem);
    }

    if (stmt->_body) {
      stmt->_body->accept(this);
    }

    irs->DBuilder.EmitBlockEnd();
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(SwitchErrorStatement *stmt) override {
    IF_LOG Logger::println("SwitchErrorStatement::toIR(): %s",
                           stmt->loc.toChars());
    LOG_SCOPE;
    assert(!irs->dcomputetarget);

    auto &PGO = irs->funcGen().pgo;
    PGO.setCurrentStmt(stmt);

    if (global.params.checkAction == CHECKACTION_C) {
      auto module = irs->func()->decl->getModule();
      DtoCAssert(module, stmt->loc, DtoConstCString("no switch default"));
      return;
    }

    // `stmt->exp` is a CallExpression to `object.__switch_error!()`
    assert(stmt->exp);
    toElemDtor(stmt->exp);

    gIR->ir->CreateUnreachable();
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(InlineAsmStatement *stmt) override {
    assert(!irs->dcomputetarget);
    AsmStatement_toIR(stmt, irs);
  }

  void visit(GccAsmStatement *stmt) override {
    assert(!irs->dcomputetarget);
    GccAsmStatement_toIR(stmt, irs);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(CompoundAsmStatement *stmt) override {
    assert(!irs->dcomputetarget);
    CompoundAsmStatement_toIR(stmt, irs);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(ImportStatement *stmt) override {
    for (auto s : *stmt->imports) {
      assert(s->isImport());
      irs->DBuilder.EmitImport(static_cast<Import *>(s));
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(Statement *stmt) override {
    error(stmt->loc, "Statement type Statement not implemented: `%s`",
          stmt->toChars());
    fatal();
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(PragmaStatement *stmt) override {
    error(stmt->loc, "Statement type PragmaStatement not implemented: `%s`",
          stmt->toChars());
    fatal();
  }
};

//////////////////////////////////////////////////////////////////////////////

void Statement_toIR(Statement *s, IRState *irs) {
  ToIRVisitor v(irs);
  s->accept(&v);
}
