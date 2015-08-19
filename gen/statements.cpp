//===-- statements.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "init.h"
#include "mars.h"
#include "module.h"
#include "mtype.h"
#include "port.h"
#include "gen/abi.h"
#include "gen/arrays.h"
#include "gen/classes.h"
#include "gen/coverage.h"
#include "gen/dvalue.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include "ir/irmodule.h"
#if LDC_LLVM_VER >= 305
#include "llvm/IR/CFG.h"
#else
#include "llvm/Support/CFG.h"
#endif
#if LDC_LLVM_VER >= 303
#include "llvm/IR/InlineAsm.h"
#else
#include "llvm/InlineAsm.h"
#endif
#include <fstream>
#include <math.h>
#include <stdio.h>

// Need to include this after the other DMD includes because of missing
// dependencies.
#include "hdrgen.h"

//////////////////////////////////////////////////////////////////////////////
// FIXME: Integrate these functions
void AsmStatement_toIR(AsmStatement *stmt, IRState * irs);
void CompoundAsmStatement_toIR(CompoundAsmStatement *stmt, IRState* p);

//////////////////////////////////////////////////////////////////////////////

// used to build the sorted list of cases
struct Case : RootObject
{
    StringExp* str;
    size_t index;

    Case(StringExp* s, size_t i) {
        str = s;
        index = i;
    }

    int compare(RootObject *obj) {
        Case* c2 = static_cast<Case*>(obj);
        return str->compare(c2->str);
    }
};

static LLValue* call_string_switch_runtime(llvm::Value* table, Expression* e)
{
    Type* dt = e->type->toBasetype();
    Type* dtnext = dt->nextOf()->toBasetype();
    TY ty = dtnext->ty;
    const char* fname;
    if (ty == Tchar) {
        fname = "_d_switch_string";
    }
    else if (ty == Twchar) {
        fname = "_d_switch_ustring";
    }
    else if (ty == Tdchar) {
        fname = "_d_switch_dstring";
    }
    else {
        llvm_unreachable("not char/wchar/dchar");
    }

    llvm::Function* fn = LLVM_D_GetRuntimeFunction(e->loc, gIR->module, fname);

    IF_LOG {
        Logger::cout() << *table->getType() << '\n';
        Logger::cout() << *fn->getFunctionType()->getParamType(0) << '\n';
    }
    assert(table->getType() == fn->getFunctionType()->getParamType(0));

    DValue* val = toElemDtor(e);
    LLValue* llval = val->getRVal();
    assert(llval->getType() == fn->getFunctionType()->getParamType(1));

    LLCallSite call = gIR->CreateCallOrInvoke(fn, table, llval);

    return call.getInstruction();
}

//////////////////////////////////////////////////////////////////////////////

class ToIRVisitor : public Visitor {
    IRState *irs;
public:

    ToIRVisitor(IRState *irs) : irs(irs) { }

    //////////////////////////////////////////////////////////////////////////

    // Import all functions from class Visitor
    using Visitor::visit;

    //////////////////////////////////////////////////////////////////////////

    void visit(CompoundStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("CompoundStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        for (Statements::iterator I = stmt->statements->begin(),
                                  E = stmt->statements->end();
                                  I != E; ++I)
        {
            Statement *s = *I;
            if (s) {
                s->accept(this);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ReturnStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("ReturnStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // emit dwarf stop point
        irs->DBuilder.EmitStopPoint(stmt->loc);

        emitCoverageLinecountInc(stmt->loc);

        // The LLVM value to return, or null for void returns.
        llvm::Value *returnValue = 0;

        // is there a return value expression?
        if (stmt->exp || (!stmt->exp && (irs->topfunc() == irs->mainFunc)) )
        {
            // if the functions return type is void this means that
            // we are returning through a pointer argument
            if (irs->topfunc()->getReturnType() == LLType::getVoidTy(irs->context()))
            {
                // sanity check
                IrFunction* f = irs->func();
                assert(getIrFunc(f->decl)->retArg);

                // FIXME: is there ever a case where a sret return needs to be rewritten for the ABI?

                // get return pointer
                DValue* rvar = new DVarValue(f->type->next, getIrFunc(f->decl)->retArg);
                DValue* e = toElemDtor(stmt->exp);
                // store return value
                if (rvar->getLVal() != e->getRVal())
                {
                    DtoAssign(stmt->loc, rvar, e, TOKblit);
                }

                // call postblit if necessary
                if (!irs->func()->type->isref && !(f->decl->nrvo_can && f->decl->nrvo_var))
                    callPostblit(stmt->loc, stmt->exp, rvar->getLVal());
            }
            // the return type is not void, so this is a normal "register" return
            else
            {
                if (!stmt->exp && (irs->topfunc() == irs->mainFunc)) {
                    returnValue = LLConstant::getNullValue(irs->mainFunc->getReturnType());
                } else {
                    if (stmt->exp->op == TOKnull)
                        stmt->exp->type = irs->func()->type->next;
                    DValue* dval = 0;
                    // call postblit if necessary
                    if (!irs->func()->type->isref) {
                        dval = toElemDtor(stmt->exp);
                        callPostblit(stmt->loc, stmt->exp, dval->getRVal());
                    } else {
                        Expression *ae = stmt->exp->addressOf();
                        dval = toElemDtor(ae);
                    }
                    // do abi specific transformations on the return value
                    returnValue = getIrFunc(irs->func()->decl)->irFty.putRet(stmt->exp->type, dval);
                }

                IF_LOG Logger::cout() << "return value is '" << returnValue << "'\n";

                IrFunction* f = irs->func();
                // Hack around LDC assuming structs and static arrays are in memory:
                // If the function returns a struct or a static array, and the return
                // value is a pointer to a struct or a static array, load from it
                // before returning.
                int ty = f->type->next->toBasetype()->ty;
                if (returnValue->getType() != irs->topfunc()->getReturnType() &&
                    (ty == Tstruct
                     || ty == Tsarray
                     ) && isaPointer(returnValue->getType()))
                {
                    Logger::println("Loading value for return");
                    returnValue = DtoLoad(returnValue);
                }

                // can happen for classes and void main
                if (returnValue->getType() != irs->topfunc()->getReturnType())
                {
                    // for the main function this only happens if it is declared as void
                    // and then contains a return (exp); statement. Since the actual
                    // return type remains i32, we just throw away the exp value
                    // and return 0 instead
                    // if we're not in main, just bitcast
                    if (irs->topfunc() == irs->mainFunc)
                        returnValue = LLConstant::getNullValue(irs->mainFunc->getReturnType());
                    else
                        returnValue = irs->ir->CreateBitCast(returnValue,
                            irs->topfunc()->getReturnType());

                    IF_LOG Logger::cout() << "return value after cast: " << *returnValue << '\n';
                }
            }
        }
        else
        {
            // no return value expression means it's a void function.
            assert(irs->topfunc()->getReturnType() == LLType::getVoidTy(irs->context()));
        }

        // If there are no cleanups to run, we try to keep the IR simple and
        // just directly emit the return instruction.

        const bool loadFromSlot = irs->func()->scopes->currentCleanupScope() != 0;
        if (loadFromSlot) {
            const bool retBlockExisted = !!irs->func()->retBlock;
            if (!retBlockExisted) {
                irs->func()->retBlock = llvm::BasicBlock::Create(
                    irs->context(), "return", irs->topfunc());
                if (returnValue) {
                    irs->func()->retValSlot = DtoRawAlloca(returnValue->getType(),
                        0, "return.slot");
                }
            }

            // Create the store to the slot at the end of our current basic
            // block, before we run the cleanups.
            if (returnValue) {
                irs->ir->CreateStore(returnValue, irs->func()->retValSlot);
            }

            // Now run the cleanups.
            irs->func()->scopes->runAllCleanups(irs->func()->retBlock);

            // If the return block already exists, we are golden. Otherwise, go
            // ahead and emit it now.
            if (retBlockExisted) {
                return;
            }

            irs->scope() = IRScope(irs->func()->retBlock);
        }

        if (returnValue) {
            // Hack: the frontend generates 'return 0;' as last statement of
            // 'void main()'. But the debug location is missing. Use the end
            // of function as debug location.
            if (irs->func()->decl->isMain() && !stmt->loc.linnum)
                irs->DBuilder.EmitStopPoint(irs->func()->decl->endloc);

            irs->ir->CreateRet(loadFromSlot ?
                DtoLoad(irs->func()->retValSlot) : returnValue);
        }
        else
        {
            irs->ir->CreateRetVoid();
        }

        // TODO: Should not be needed
        llvm::BasicBlock* bb = llvm::BasicBlock::Create(gIR->context(), "afterreturn", irs->topfunc());
        irs->scope() = IRScope(bb);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ExpStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("ExpStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // emit dwarf stop point
        irs->DBuilder.EmitStopPoint(stmt->loc);

        emitCoverageLinecountInc(stmt->loc);

        if (stmt->exp) {
            elem* e;
            // a cast(void) around the expression is allowed, but doesn't require any code
            if (stmt->exp->op == TOKcast && stmt->exp->type == Type::tvoid) {
                CastExp* cexp = static_cast<CastExp*>(stmt->exp);
                e = toElemDtor(cexp->e1);
            }
            else
                e = toElemDtor(stmt->exp);
            delete e;
        }
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(IfStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("IfStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // start a dwarf lexical block
        irs->DBuilder.EmitBlockStart(stmt->loc);
        emitCoverageLinecountInc(stmt->loc);

        if (stmt->match)
            DtoRawVarDeclaration(stmt->match);

        DValue* cond_e = toElemDtor(stmt->condition);
        LLValue* cond_val = cond_e->getRVal();

        llvm::BasicBlock* ifbb = llvm::BasicBlock::Create(irs->context(), "if", irs->topfunc());
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(irs->context(), "endif", irs->topfunc());
        llvm::BasicBlock* elsebb = stmt->elsebody ? llvm::BasicBlock::Create(irs->context(), "else", irs->topfunc(), endbb) : endbb;

        if (cond_val->getType() != LLType::getInt1Ty(irs->context())) {
            IF_LOG Logger::cout() << "if conditional: " << *cond_val << '\n';
            cond_val = DtoCast(stmt->loc, cond_e, Type::tbool)->getRVal();
        }
        llvm::BranchInst::Create(ifbb, elsebb, cond_val, irs->scopebb());

        // replace current scope
        irs->scope() = IRScope(ifbb);

        // do scoped statements

        if (stmt->ifbody) {
            irs->DBuilder.EmitBlockStart(stmt->ifbody->loc);
            stmt->ifbody->accept(this);
            irs->DBuilder.EmitBlockEnd();
        }
        if (!irs->scopereturned()) {
            llvm::BranchInst::Create(endbb, irs->scopebb());
        }

        if (stmt->elsebody) {
            irs->scope() = IRScope(elsebb);
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
        irs->scope() = IRScope(endbb);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ScopeStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("ScopeStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        if (stmt->statement) {
            irs->DBuilder.EmitBlockStart(stmt->statement->loc);
            stmt->statement->accept(this);
            irs->DBuilder.EmitBlockEnd();
        }
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(WhileStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("WhileStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // start a dwarf lexical block
        irs->DBuilder.EmitBlockStart(stmt->loc);

        // create while blocks

        llvm::BasicBlock* whilebb = llvm::BasicBlock::Create(irs->context(), "whilecond", irs->topfunc());
        llvm::BasicBlock* whilebodybb = llvm::BasicBlock::Create(irs->context(), "whilebody", irs->topfunc());
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(irs->context(), "endwhile", irs->topfunc());

        // move into the while block
        irs->ir->CreateBr(whilebb);

        // replace current scope
        irs->scope() = IRScope(whilebb);

        // create the condition
        emitCoverageLinecountInc(stmt->condition->loc);
        DValue* cond_e = toElemDtor(stmt->condition);
        LLValue* cond_val = DtoCast(stmt->loc, cond_e, Type::tbool)->getRVal();
        delete cond_e;

        // conditional branch
        llvm::BranchInst::Create(whilebodybb, endbb, cond_val, irs->scopebb());

        // rewrite scope
        irs->scope() = IRScope(whilebodybb);

        // while body code
        irs->func()->scopes->pushLoopTarget(stmt, whilebb, endbb);
        if (stmt->body)
            stmt->body->accept(this);
        irs->func()->scopes->popLoopTarget();

        // loop
        if (!irs->scopereturned())
            llvm::BranchInst::Create(whilebb, irs->scopebb());

        // rewrite the scope
        irs->scope() = IRScope(endbb);

        // end the dwarf lexical block
        irs->DBuilder.EmitBlockEnd();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(DoStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("DoStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // start a dwarf lexical block
        irs->DBuilder.EmitBlockStart(stmt->loc);

        // create while blocks
        llvm::BasicBlock* dowhilebb = llvm::BasicBlock::Create(irs->context(), "dowhile", irs->topfunc());
        llvm::BasicBlock* condbb = llvm::BasicBlock::Create(irs->context(), "dowhilecond", irs->topfunc());
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(irs->context(), "enddowhile", irs->topfunc());

        // move into the while block
        assert(!irs->scopereturned());
        llvm::BranchInst::Create(dowhilebb, irs->scopebb());

        // replace current scope
        irs->scope() = IRScope(dowhilebb);

        // do-while body code
        irs->func()->scopes->pushLoopTarget(stmt, condbb, endbb);
        if (stmt->body)
            stmt->body->accept(this);
        irs->func()->scopes->popLoopTarget();

        // branch to condition block
        llvm::BranchInst::Create(condbb, irs->scopebb());
        irs->scope() = IRScope(condbb);

        // create the condition
        emitCoverageLinecountInc(stmt->condition->loc);
        DValue* cond_e = toElemDtor(stmt->condition);
        LLValue* cond_val = DtoCast(stmt->loc, cond_e, Type::tbool)->getRVal();
        delete cond_e;

        // conditional branch
        llvm::BranchInst::Create(dowhilebb, endbb, cond_val, irs->scopebb());

        // rewrite the scope
        irs->scope() = IRScope(endbb);

        // end the dwarf lexical block
        irs->DBuilder.EmitBlockEnd();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ForStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("ForStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // start new dwarf lexical block
        irs->DBuilder.EmitBlockStart(stmt->loc);

        // create for blocks
        llvm::BasicBlock* forbb = llvm::BasicBlock::Create(irs->context(), "forcond", irs->topfunc());
        llvm::BasicBlock* forbodybb = llvm::BasicBlock::Create(irs->context(), "forbody", irs->topfunc());
        llvm::BasicBlock* forincbb = llvm::BasicBlock::Create(irs->context(), "forinc", irs->topfunc());
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(irs->context(), "endfor", irs->topfunc());

        // init
        if (stmt->init != 0)
            stmt->init->accept(this);

        // move into the for condition block, ie. start the loop
        assert(!irs->scopereturned());
        llvm::BranchInst::Create(forbb, irs->scopebb());

        // In case of loops that have been rewritten to a composite statement
        // containing the initializers and then the actual loop, we need to
        // register the former as target scope start.
        Statement* scopeStart = stmt->getRelatedLabeled();
        while (ScopeStatement* scope = scopeStart->isScopeStatement())
        {
            scopeStart = scope->statement;
        }
        irs->func()->scopes->pushLoopTarget(scopeStart, forincbb, endbb);

        // replace current scope
        irs->scope() = IRScope(forbb);

        // create the condition
        llvm::Value* cond_val;
        if (stmt->condition)
        {
            emitCoverageLinecountInc(stmt->condition->loc);
            DValue* cond_e = toElemDtor(stmt->condition);
            cond_val = DtoCast(stmt->loc, cond_e, Type::tbool)->getRVal();
            delete cond_e;
        }
        else
        {
            cond_val = DtoConstBool(true);
        }

        // conditional branch
        assert(!irs->scopereturned());
        llvm::BranchInst::Create(forbodybb, endbb, cond_val, irs->scopebb());

        // rewrite scope
        irs->scope() = IRScope(forbodybb);

        // do for body code
        if (stmt->body)
            stmt->body->accept(this);

        // move into the for increment block
        if (!irs->scopereturned())
            llvm::BranchInst::Create(forincbb, irs->scopebb());
        irs->scope() = IRScope(forincbb);

        // increment
        if (stmt->increment) {
            emitCoverageLinecountInc(stmt->increment->loc);
            DValue* inc = toElemDtor(stmt->increment);
            delete inc;
        }

        // loop
        if (!irs->scopereturned())
            llvm::BranchInst::Create(forbb, irs->scopebb());

        irs->func()->scopes->popLoopTarget();

        // rewrite the scope
        irs->scope() = IRScope(endbb);

        // end the dwarf lexical block
        irs->DBuilder.EmitBlockEnd();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(BreakStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("BreakStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // don't emit two terminators in a row
        // happens just before DMD generated default statements if the last case terminates
        if (irs->scopereturned())
            return;

        // emit dwarf stop point
        irs->DBuilder.EmitStopPoint(stmt->loc);

        emitCoverageLinecountInc(stmt->loc);

        if (stmt->ident) {
            IF_LOG Logger::println("ident = %s", stmt->ident->toChars());

            // Get the loop or break statement the label refers to
            Statement* targetStatement = stmt->target->statement;
            ScopeStatement* tmp;
            while((tmp = targetStatement->isScopeStatement()))
                targetStatement = tmp->statement;

            irs->func()->scopes->breakToStatement(targetStatement);
        } else {
            irs->func()->scopes->breakToClosest();
        }

        // the break terminated this basicblock, start a new one
        llvm::BasicBlock* bb = llvm::BasicBlock::Create(irs->context(), "afterbreak", irs->topfunc());
        irs->scope() = IRScope(bb);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ContinueStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("ContinueStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // emit dwarf stop point
        irs->DBuilder.EmitStopPoint(stmt->loc);

        emitCoverageLinecountInc(stmt->loc);

        if (stmt->ident) {
            IF_LOG Logger::println("ident = %s", stmt->ident->toChars());

            // get the loop statement the label refers to
            Statement* targetLoopStatement = stmt->target->statement;
            ScopeStatement* tmp;
            while((tmp = targetLoopStatement->isScopeStatement()))
                targetLoopStatement = tmp->statement;

            irs->func()->scopes->continueWithLoop(targetLoopStatement);
        } else {
            irs->func()->scopes->continueWithClosest();
        }

        // the break terminated this basicblock, start a new one
        llvm::BasicBlock* bb = llvm::BasicBlock::Create(irs->context(), "afterbreak", irs->topfunc());
        irs->scope() = IRScope(bb);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(OnScopeStatement *stmt) LLVM_OVERRIDE {
        stmt->error("Internal Compiler Error: OnScopeStatement should have been lowered by frontend.");
        fatal();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(TryFinallyStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("TryFinallyStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // emit dwarf stop point
        irs->DBuilder.EmitStopPoint(stmt->loc);

        // We only need to consider exception handling/cleanup issues if there
        // is both a try and a finally block. If not, just directly emit what
        // is present.
        if (!stmt->body || !stmt->finalbody) {
            if (stmt->body) {
                irs->DBuilder.EmitBlockStart(stmt->body->loc);
                stmt->body->accept(this);
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
        llvm::BasicBlock* trybb = irs->scopebb();

        // Emit the finally block and set up the cleanup scope for it.
        llvm::BasicBlock* finallybb =
            llvm::BasicBlock::Create(irs->context(), "finally", irs->topfunc());
        irs->scope() = IRScope(finallybb);
        irs->DBuilder.EmitBlockStart(stmt->finalbody->loc);
        stmt->finalbody->accept(this);
        irs->DBuilder.EmitBlockEnd();

        CleanupCursor cleanupBefore = irs->func()->scopes->currentCleanupScope();
        irs->func()->scopes->pushCleanup(finallybb, irs->scopebb());

        // Emit the try block.
        irs->scope() = IRScope(trybb);

        assert(stmt->body);
        irs->DBuilder.EmitBlockStart(stmt->body->loc);
        stmt->body->accept(this);
        irs->DBuilder.EmitBlockEnd();

        // Create a block to branch to after successfully running the try block
        // and any cleanups.
        if (!irs->scopereturned()) {
            llvm::BasicBlock* successbb = llvm::BasicBlock::Create(irs->context(),
                "try.success", irs->topfunc());
            irs->func()->scopes->runCleanups(cleanupBefore, successbb);
            irs->scope() = IRScope(successbb);
        }
        irs->func()->scopes->popCleanups(cleanupBefore);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(TryCatchStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("TryCatchStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // Emit dwarf stop point
        irs->DBuilder.EmitStopPoint(stmt->loc);

        // We'll append the "try" part to the current basic block later. No need
        // for an extra one (we'd need to branch to it unconditionally anyway).
        llvm::BasicBlock* trybb = irs->scopebb();

        // Create a basic block to branch to after leaving the try or an
        // associated catch block successfully.
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(irs->context(),
            "try.success.or.caught", irs->topfunc());

        assert(stmt->catches);

        for (Catches::reverse_iterator it = stmt->catches->rbegin(),
                                       end = stmt->catches->rend();
             it != end; ++it
        ) {
            llvm::BasicBlock* catchBlock = llvm::BasicBlock::Create(irs->context(),
                llvm::Twine("catch.") + (*it)->type->toChars(),
                irs->topfunc(), endbb);

            irs->scope() = IRScope(catchBlock);
            irs->DBuilder.EmitBlockStart((*it)->loc);

            llvm::Function* enterCatchFn =
                LLVM_D_GetRuntimeFunction(Loc(), irs->module, "_d_eh_enter_catch");
            irs->ir->CreateCall(enterCatchFn);

            // For catches that use the Throwable object, create storage for it.
            // We will set it in the code that branches from the landing pads
            // (there might be more than one) to catchBlock.
            if ((*it)->var) {
                llvm::Value* ehPtr = irs->func()->getOrCreateEhPtrSlot();

#if LDC_LLVM_VER >= 305
                if (!global.params.targetTriple.isWindowsMSVCEnvironment())
#endif
                {
                    // ehPtr is a pointer to _d_exception, which has a reference
                    // to the Throwable object at offset 0.
                    ehPtr = irs->ir->CreateLoad(ehPtr);
                }

                llvm::Type* llCatchVarType = DtoType((*it)->var->type); // e.g., Throwable*

                // Use the same storage for all exceptions that are not accessed in
                // nested functions
                if (!(*it)->var->nestedrefs.dim) {
                    assert(!isIrLocalCreated((*it)->var));
                    IrLocal* irLocal = getIrLocal((*it)->var, true);
                    irLocal->value = DtoBitCast(ehPtr, getPtrToType(llCatchVarType));
                } else {
                    // This will alloca if we haven't already and take care of nested refs
                    DtoDeclarationExp((*it)->var);
                    IrLocal* irLocal = getIrLocal((*it)->var);

                    // Copy the exception reference over from ehPtr
                    llvm::Value* exc = DtoLoad(DtoBitCast(ehPtr, llCatchVarType->getPointerTo()));
                    DtoStore(exc, irLocal->value);
                }
            }

            // emit handler, if there is one
            // handler is zero for instance for 'catch { debug foo(); }'
            if ((*it)->handler) {
                Statement_toIR((*it)->handler, irs);
            }

            if (!irs->scopereturned()) {
                irs->ir->CreateBr(endbb);
            }

            irs->DBuilder.EmitBlockEnd();

            ClassDeclaration* catchType =
                (*it)->type->toBasetype()->isClassHandle();
            DtoResolveClass(catchType);

            irs->func()->scopes->pushCatch(
                getIrAggr(catchType)->getClassInfoSymbol(), catchBlock);
        }

        // Emit the try block.
        irs->scope() = IRScope(trybb);

        assert(stmt->body);
        irs->DBuilder.EmitBlockStart(stmt->body->loc);
        stmt->body->accept(this);
        irs->DBuilder.EmitBlockEnd();

        if (!irs->scopereturned()) {
            llvm::BranchInst::Create(endbb, irs->scopebb());
        }

        // Now that we have done the try block, remove the catches and continue
        // codegen in the end block the try and all the catches branch to.
        for (Catches::reverse_iterator it = stmt->catches->rbegin(),
                                       end = stmt->catches->rend();
             it != end; ++it
        ) {
            irs->func()->scopes->popCatch();
        }

        irs->scope() = IRScope(endbb);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ThrowStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("ThrowStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // emit dwarf stop point
        irs->DBuilder.EmitStopPoint(stmt->loc);

        emitCoverageLinecountInc(stmt->loc);

        assert(stmt->exp);
        DValue* e = toElemDtor(stmt->exp);

        llvm::Function* fn = LLVM_D_GetRuntimeFunction(stmt->loc, irs->module, "_d_throw_exception");
        LLValue* arg = DtoBitCast(e->getRVal(), fn->getFunctionType()->getParamType(0));;
        irs->CreateCallOrInvoke(fn, arg);
        irs->ir->CreateUnreachable();

        // TODO: Should not be needed.
        llvm::BasicBlock* bb = llvm::BasicBlock::Create(irs->context(), "afterthrow", irs->topfunc());
        irs->scope() = IRScope(bb);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(SwitchStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("SwitchStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // emit dwarf stop point
        irs->DBuilder.EmitStopPoint(stmt->loc);

        emitCoverageLinecountInc(stmt->loc);

        llvm::BasicBlock* oldbb = irs->scopebb();

        // If one of the case expressions is non-constant, we can't use
        // 'switch' instruction (that can happen because D2 allows to
        // initialize a global variable in a static constructor).
        bool useSwitchInst = true;
        for (CaseStatements::iterator I = stmt->cases->begin(),
                                      E = stmt->cases->end();
                                      I != E; ++I)
        {
            CaseStatement *cs = *I;
            VarDeclaration* vd = 0;
            if (cs->exp->op == TOKvar)
                vd = static_cast<VarExp*>(cs->exp)->var->isVarDeclaration();
            if (vd && (!vd->init || !vd->isConst())) {
                cs->llvmIdx = toElemDtor(cs->exp)->getRVal();
                useSwitchInst = false;
            }
        }

        // body block.
        // FIXME: that block is never used
        llvm::BasicBlock* bodybb = llvm::BasicBlock::Create(irs->context(), "switchbody", irs->topfunc());

        // default
        llvm::BasicBlock* defbb = 0;
        if (stmt->sdefault) {
            Logger::println("has default");
            defbb = llvm::BasicBlock::Create(irs->context(), "default", irs->topfunc());
            stmt->sdefault->bodyBB = defbb;
        }

        // end (break point)
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(irs->context(), "switchend", irs->topfunc());

        // do switch body
        assert(stmt->body);
        irs->scope() = IRScope(bodybb);
        irs->func()->scopes->pushBreakTarget(stmt, endbb);
        stmt->body->accept(this);
        irs->func()->scopes->popBreakTarget();
        if (!irs->scopereturned())
            llvm::BranchInst::Create(endbb, irs->scopebb());

        irs->scope() = IRScope(oldbb);
        if (useSwitchInst)
        {
            // string switch?
            llvm::Value* switchTable = 0;
            Objects caseArray;
            if (!stmt->condition->type->isintegral())
            {
                Logger::println("is string switch");
                // build array of the stringexpS
                caseArray.reserve(stmt->cases->dim);
                for (unsigned i=0; i < stmt->cases->dim; ++i)
                {
                    CaseStatement* cs = static_cast<CaseStatement*>(stmt->cases->data[i]);

                    assert(cs->exp->op == TOKstring);
                    caseArray.push(new Case(static_cast<StringExp*>(cs->exp), i));
                }
                // first sort it
                caseArray.sort();
                // iterate and add indices to cases
                std::vector<llvm::Constant*> inits(caseArray.dim, 0);
                for (size_t i=0; i < caseArray.dim; ++i)
                {
                    Case* c = static_cast<Case*>(caseArray.data[i]);
                    CaseStatement* cs = static_cast<CaseStatement*>(stmt->cases->data[c->index]);
                    cs->llvmIdx = DtoConstUint(i);
                    inits[i] = toConstElem(c->str, irs);
                }
                // build static array for ptr or final array
                llvm::Type* elemTy = DtoType(stmt->condition->type);
                LLArrayType* arrTy = llvm::ArrayType::get(elemTy, inits.size());
                LLConstant* arrInit = LLConstantArray::get(arrTy, inits);
                LLGlobalVariable* arr = new llvm::GlobalVariable(irs->module, arrTy, true, llvm::GlobalValue::InternalLinkage, arrInit, ".string_switch_table_data");

                LLType* elemPtrTy = getPtrToType(elemTy);
                LLConstant* arrPtr = llvm::ConstantExpr::getBitCast(arr, elemPtrTy);

                // build the static table
                LLType* types[] = { DtoSize_t(), elemPtrTy };
                LLStructType* sTy = llvm::StructType::get(irs->context(), types, false);
                LLConstant* sinits[] = { DtoConstSize_t(inits.size()), arrPtr };
                switchTable = llvm::ConstantStruct::get(sTy, llvm::ArrayRef<LLConstant*>(sinits));
            }

            // condition var
            LLValue* condVal;
            // integral switch
            if (stmt->condition->type->isintegral()) {
                DValue* cond = toElemDtor(stmt->condition);
                condVal = cond->getRVal();
            }
            // string switch
            else {
                condVal = call_string_switch_runtime(switchTable, stmt->condition);
            }

            // create switch and add the cases
            llvm::SwitchInst* si = llvm::SwitchInst::Create(condVal, defbb ? defbb : endbb, stmt->cases->dim, irs->scopebb());
            for (CaseStatements::iterator I = stmt->cases->begin(),
                                          E = stmt->cases->end();
                                          I != E; ++I)
            {
                CaseStatement *cs = *I;
                si->addCase(isaConstantInt(cs->llvmIdx), cs->bodyBB);
            }
        }
        else
        { // we can't use switch, so we will use a bunch of br instructions instead
            DValue* cond = toElemDtor(stmt->condition);
            LLValue *condVal = cond->getRVal();

            llvm::BasicBlock* nextbb = llvm::BasicBlock::Create(irs->context(), "checkcase", irs->topfunc());
            llvm::BranchInst::Create(nextbb, irs->scopebb());

            irs->scope() = IRScope(nextbb);
            for (CaseStatements::iterator I = stmt->cases->begin(),
                                          E = stmt->cases->end();
                                          I != E; ++I)
            {
                CaseStatement *cs = *I;

                LLValue *cmp = irs->ir->CreateICmp(llvm::ICmpInst::ICMP_EQ, cs->llvmIdx, condVal, "checkcase");
                nextbb = llvm::BasicBlock::Create(irs->context(), "checkcase", irs->topfunc());
                llvm::BranchInst::Create(cs->bodyBB, nextbb, cmp, irs->scopebb());
                irs->scope() = IRScope(nextbb);
            }

            if (stmt->sdefault) {
                llvm::BranchInst::Create(stmt->sdefault->bodyBB, irs->scopebb());
            } else {
                llvm::BranchInst::Create(endbb, irs->scopebb());
            }
            endbb->moveAfter(nextbb);
        }

        irs->scope() = IRScope(endbb);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(CaseStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("CaseStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        llvm::BasicBlock* nbb = llvm::BasicBlock::Create(irs->context(), "case", irs->topfunc());
        if (stmt->bodyBB && !stmt->bodyBB->getTerminator())
        {
            llvm::BranchInst::Create(nbb, stmt->bodyBB);
        }
        stmt->bodyBB = nbb;

        if (stmt->llvmIdx == NULL) {
            llvm::Constant *c = toConstElem(stmt->exp, irs);
            stmt->llvmIdx = isaConstantInt(c);
        }

        if (!irs->scopereturned())
            llvm::BranchInst::Create(stmt->bodyBB, irs->scopebb());

        irs->scope() = IRScope(stmt->bodyBB);

        assert(stmt->statement);
        irs->DBuilder.EmitBlockStart(stmt->statement->loc);
        emitCoverageLinecountInc(stmt->loc);
        stmt->statement->accept(this);
        irs->DBuilder.EmitBlockEnd();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(DefaultStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("DefaultStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        assert(stmt->bodyBB);

        llvm::BasicBlock* nbb = llvm::BasicBlock::Create(irs->context(), "default", irs->topfunc());

        if (!stmt->bodyBB->getTerminator())
        {
            llvm::BranchInst::Create(nbb, stmt->bodyBB);
        }
        stmt->bodyBB = nbb;

        if (!irs->scopereturned())
            llvm::BranchInst::Create(stmt->bodyBB, irs->scopebb());

        irs->scope() = IRScope(stmt->bodyBB);

        assert(stmt->statement);
        irs->DBuilder.EmitBlockStart(stmt->statement->loc);
        emitCoverageLinecountInc(stmt->loc);
        stmt->statement->accept(this);
        irs->DBuilder.EmitBlockEnd();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(UnrolledLoopStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("UnrolledLoopStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // if no statements, there's nothing to do
        if (!stmt->statements || !stmt->statements->dim)
            return;

        // start a dwarf lexical block
        irs->DBuilder.EmitBlockStart(stmt->loc);

        // DMD doesn't fold stuff like continue/break, and since this isn't really a loop
        // we have to keep track of each statement and jump to the next/end on continue/break

        // create a block for each statement
        size_t nstmt = stmt->statements->dim;
        llvm::SmallVector<llvm::BasicBlock*, 4> blocks(nstmt, NULL);

        for (size_t i=0; i < nstmt; i++)
        {
            blocks[i] = llvm::BasicBlock::Create(irs->context(), "unrolledstmt", irs->topfunc());
        }

        // create end block
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(irs->context(), "unrolledend", irs->topfunc());

        // enter first stmt
        if (!irs->scopereturned())
            irs->ir->CreateBr(blocks[0]);

        // do statements
        Statement** stmts = static_cast<Statement **>(stmt->statements->data);

        for (size_t i=0; i < nstmt; i++)
        {
            Statement *s = stmts[i];

            // get blocks
            llvm::BasicBlock* thisbb = blocks[i];
            llvm::BasicBlock* nextbb = (i+1 == nstmt) ? endbb : blocks[i+1];

            // update scope
            irs->scope() = IRScope(thisbb);

            // push loop scope
            // continue goes to next statement, break goes to end
            irs->func()->scopes->pushLoopTarget(stmt, nextbb, endbb);

            // do statement
            s->accept(this);

            // pop loop scope
            irs->func()->scopes->popLoopTarget();

            // next stmt
            if (!irs->scopereturned())
                irs->ir->CreateBr(nextbb);
        }

        // finish scope
        if (!irs->scopereturned())
            irs->ir->CreateBr(endbb);
        irs->scope() = IRScope(endbb);

        // end the dwarf lexical block
        irs->DBuilder.EmitBlockEnd();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ForeachStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("ForeachStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // start a dwarf lexical block
        irs->DBuilder.EmitBlockStart(stmt->loc);

        //assert(arguments->dim == 1);
        assert(stmt->value != 0);
        assert(stmt->aggr != 0);
        assert(stmt->func != 0);

        //Argument* arg = static_cast<Argument*>(arguments->data[0]);
        //Logger::println("Argument is %s", arg->toChars());

        IF_LOG Logger::println("aggr = %s", stmt->aggr->toChars());

        // key
        LLType* keytype = stmt->key ? DtoType(stmt->key->type) : DtoSize_t();
        LLValue* keyvar;
        if (stmt->key)
            keyvar = DtoRawVarDeclaration(stmt->key);
        else
            keyvar = DtoRawAlloca(keytype, 0, "foreachkey"); // FIXME: align?
        LLValue* zerokey = LLConstantInt::get(keytype, 0, false);

        // value
        IF_LOG Logger::println("value = %s", stmt->value->toPrettyChars());
        LLValue* valvar = NULL;
        if (!stmt->value->isRef() && !stmt->value->isOut()) {
            // Create a local variable to serve as the value.
            DtoRawVarDeclaration(stmt->value);
            valvar = getIrLocal(stmt->value)->value;
        }

        // what to iterate
        DValue* aggrval = toElemDtor(stmt->aggr);

        // get length and pointer
        LLValue* niters = DtoArrayLen(aggrval);
        LLValue* val = DtoArrayPtr(aggrval);

        if (niters->getType() != keytype)
        {
            size_t sz1 = getTypeBitSize(niters->getType());
            size_t sz2 = getTypeBitSize(keytype);
            if (sz1 < sz2)
                niters = irs->ir->CreateZExt(niters, keytype, "foreachtrunckey");
            else if (sz1 > sz2)
                niters = irs->ir->CreateTrunc(niters, keytype, "foreachtrunckey");
            else
                niters = irs->ir->CreateBitCast(niters, keytype, "foreachtrunckey");
        }

        if (stmt->op == TOKforeach) {
            new llvm::StoreInst(zerokey, keyvar, irs->scopebb());
        }
        else {
            new llvm::StoreInst(niters, keyvar, irs->scopebb());
        }

        llvm::BasicBlock* condbb = llvm::BasicBlock::Create(irs->context(), "foreachcond", irs->topfunc());
        llvm::BasicBlock* bodybb = llvm::BasicBlock::Create(irs->context(), "foreachbody", irs->topfunc());
        llvm::BasicBlock* nextbb = llvm::BasicBlock::Create(irs->context(), "foreachnext", irs->topfunc());
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(irs->context(), "foreachend", irs->topfunc());

        llvm::BranchInst::Create(condbb, irs->scopebb());

        // condition
        irs->scope() = IRScope(condbb);

        LLValue* done = 0;
        LLValue* load = DtoLoad(keyvar);
        if (stmt->op == TOKforeach) {
            done = irs->ir->CreateICmpULT(load, niters);
        }
        else if (stmt->op == TOKforeach_reverse) {
            done = irs->ir->CreateICmpUGT(load, zerokey);
            load = irs->ir->CreateSub(load, LLConstantInt::get(keytype, 1, false));
            DtoStore(load, keyvar);
        }
        llvm::BranchInst::Create(bodybb, endbb, done, irs->scopebb());

        // init body
        irs->scope() = IRScope(bodybb);

        // get value for this iteration
        LLValue* loadedKey = irs->ir->CreateLoad(keyvar);
        LLValue* gep = DtoGEP1(val, loadedKey);

        if (!stmt->value->isRef() && !stmt->value->isOut()) {
            // Copy value to local variable, and use it as the value variable.
            DVarValue dst(stmt->value->type, valvar);
            DVarValue src(stmt->value->type, gep);
            DtoAssign(stmt->loc, &dst, &src);
            getIrLocal(stmt->value)->value = valvar;
        } else {
            // Use the GEP as the address of the value variable.
            DtoRawVarDeclaration(stmt->value, gep);
        }

        // emit body
        irs->func()->scopes->pushLoopTarget(stmt, nextbb, endbb);
        if (stmt->body)
            stmt->body->accept(this);
        irs->func()->scopes->popLoopTarget();

        if (!irs->scopereturned())
            llvm::BranchInst::Create(nextbb, irs->scopebb());

        // next
        irs->scope() = IRScope(nextbb);
        if (stmt->op == TOKforeach) {
            LLValue* load = DtoLoad(keyvar);
            load = irs->ir->CreateAdd(load, LLConstantInt::get(keytype, 1, false));
            DtoStore(load, keyvar);
        }
        llvm::BranchInst::Create(condbb, irs->scopebb());

        // end the dwarf lexical block
        irs->DBuilder.EmitBlockEnd();

        // end
        irs->scope() = IRScope(endbb);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ForeachRangeStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("ForeachRangeStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // start a dwarf lexical block
        irs->DBuilder.EmitBlockStart(stmt->loc);

        // evaluate lwr/upr
        assert(stmt->lwr->type->isintegral());
        LLValue* lower = toElemDtor(stmt->lwr)->getRVal();
        assert(stmt->upr->type->isintegral());
        LLValue* upper = toElemDtor(stmt->upr)->getRVal();

        // handle key
        assert(stmt->key->type->isintegral());
        LLValue* keyval = DtoRawVarDeclaration(stmt->key);

        // store initial value in key
        if (stmt->op == TOKforeach)
            DtoStore(lower, keyval);
        else
            DtoStore(upper, keyval);

        // set up the block we'll need
        llvm::BasicBlock* condbb = llvm::BasicBlock::Create(irs->context(), "foreachrange_cond", irs->topfunc());
        llvm::BasicBlock* bodybb = llvm::BasicBlock::Create(irs->context(), "foreachrange_body", irs->topfunc());
        llvm::BasicBlock* nextbb = llvm::BasicBlock::Create(irs->context(), "foreachrange_next", irs->topfunc());
        llvm::BasicBlock* endbb = llvm::BasicBlock::Create(irs->context(), "foreachrange_end", irs->topfunc());

        // jump to condition
        llvm::BranchInst::Create(condbb, irs->scopebb());

        // CONDITION
        irs->scope() = IRScope(condbb);

        // first we test that lwr < upr
        lower = DtoLoad(keyval);
        assert(lower->getType() == upper->getType());
        llvm::ICmpInst::Predicate cmpop;
        if (isLLVMUnsigned(stmt->key->type))
        {
            cmpop = (stmt->op == TOKforeach)
            ? llvm::ICmpInst::ICMP_ULT
            : llvm::ICmpInst::ICMP_UGT;
        }
        else
        {
            cmpop = (stmt->op == TOKforeach)
            ? llvm::ICmpInst::ICMP_SLT
            : llvm::ICmpInst::ICMP_SGT;
        }
        LLValue* cond = irs->ir->CreateICmp(cmpop, lower, upper);

        // jump to the body if range is ok, to the end if not
        llvm::BranchInst::Create(bodybb, endbb, cond, irs->scopebb());

        // BODY
        irs->scope() = IRScope(bodybb);

        // reverse foreach decrements here
        if (stmt->op == TOKforeach_reverse)
        {
            LLValue* v = DtoLoad(keyval);
            LLValue* one = LLConstantInt::get(v->getType(), 1, false);
            v = irs->ir->CreateSub(v, one);
            DtoStore(v, keyval);
        }

        // emit body
        irs->func()->scopes->pushLoopTarget(stmt, nextbb, endbb);
        if (stmt->body)
            stmt->body->accept(this);
        irs->func()->scopes->popLoopTarget();

        // jump to next iteration
        if (!irs->scopereturned())
            llvm::BranchInst::Create(nextbb, irs->scopebb());

        // NEXT
        irs->scope() = IRScope(nextbb);

        // forward foreach increments here
        if (stmt->op == TOKforeach)
        {
            LLValue* v = DtoLoad(keyval);
            LLValue* one = LLConstantInt::get(v->getType(), 1, false);
            v = irs->ir->CreateAdd(v, one);
            DtoStore(v, keyval);
        }

        // jump to condition
        llvm::BranchInst::Create(condbb, irs->scopebb());

        // end the dwarf lexical block
        irs->DBuilder.EmitBlockEnd();

        // END
        irs->scope() = IRScope(endbb);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(LabelStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("LabelStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        // if it's an inline asm label, we don't create a basicblock, just emit it in the asm
        if (irs->asmBlock)
        {
            IRAsmStmt* a = new IRAsmStmt;
            std::stringstream label;
            printLabelName(label, mangleExact(irs->func()->decl), stmt->ident->toChars());
            label << ":";
            a->code = label.str();
            irs->asmBlock->s.push_back(a);
            irs->asmBlock->internalLabels.push_back(stmt->ident);

            // disable inlining
            irs->func()->setNeverInline();
        }
        else
        {
            llvm::BasicBlock* labelBB = llvm::BasicBlock::Create(irs->context(),
                llvm::Twine("label.") + stmt->ident->toChars(), irs->topfunc());
            irs->func()->scopes->addLabelTarget(stmt->ident, labelBB);

            if (!irs->scopereturned())
                llvm::BranchInst::Create(labelBB, irs->scopebb());

            irs->scope() = IRScope(labelBB);
        }

        if (stmt->statement) {
            stmt->statement->accept(this);
        }
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(GotoStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("GotoStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        irs->DBuilder.EmitStopPoint(stmt->loc);

        emitCoverageLinecountInc(stmt->loc);

        DtoGoto(stmt->loc, stmt->label);

        // TODO: Should not be needed.
        llvm::BasicBlock* bb = llvm::BasicBlock::Create(irs->context(), "aftergoto", irs->topfunc());
        irs->scope() = IRScope(bb);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(GotoDefaultStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("GotoDefaultStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        irs->DBuilder.EmitStopPoint(stmt->loc);

        emitCoverageLinecountInc(stmt->loc);

        assert(!irs->scopereturned());
        assert(stmt->sw->sdefault->bodyBB);

#if 0
        // TODO: Store switch scopes.
        DtoEnclosingHandlers(stmt->loc, stmt->sw);
#endif

        llvm::BranchInst::Create(stmt->sw->sdefault->bodyBB, irs->scopebb());

        // TODO: Should not be needed.
        llvm::BasicBlock* bb = llvm::BasicBlock::Create(irs->context(), "aftergotodefault", irs->topfunc());
        irs->scope() = IRScope(bb);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(GotoCaseStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("GotoCaseStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        irs->DBuilder.EmitStopPoint(stmt->loc);

        emitCoverageLinecountInc(stmt->loc);

        assert(!irs->scopereturned());
        if (!stmt->cs->bodyBB)
        {
            stmt->cs->bodyBB = llvm::BasicBlock::Create(irs->context(), "goto_case", irs->topfunc());
        }

#if 0
        // TODO: Store switch scopes.
        DtoEnclosingHandlers(stmt->loc, stmt->sw);
#endif

        llvm::BranchInst::Create(stmt->cs->bodyBB, irs->scopebb());

        // TODO: Should not be needed.
        llvm::BasicBlock* bb = llvm::BasicBlock::Create(irs->context(), "aftergotocase", irs->topfunc());
        irs->scope() = IRScope(bb);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(WithStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("WithStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        irs->DBuilder.EmitBlockStart(stmt->loc);

        assert(stmt->exp);

        // with(..) can either be used with expressions or with symbols
        // wthis == null indicates the symbol form
        if (stmt->wthis) {
            DValue* e = toElemDtor(stmt->exp);
            LLValue* mem = DtoRawVarDeclaration(stmt->wthis);
            DtoStore(e->getRVal(), mem);
        }

        if (stmt->body)
            stmt->body->accept(this);

        irs->DBuilder.EmitBlockEnd();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(SwitchErrorStatement *stmt) LLVM_OVERRIDE {
        IF_LOG Logger::println("SwitchErrorStatement::toIR(): %s", stmt->loc.toChars());
        LOG_SCOPE;

        llvm::Function* fn = LLVM_D_GetRuntimeFunction(stmt->loc, irs->module, "_d_switch_error");

        LLValue *moduleInfoSymbol = getIrModule(irs->func()->decl->getModule())->moduleInfoSymbol();
        LLType *moduleInfoType = DtoType(Module::moduleinfo->type);

        LLCallSite call = irs->CreateCallOrInvoke(
            fn,
            DtoBitCast(moduleInfoSymbol, getPtrToType(moduleInfoType)),
            DtoConstUint(stmt->loc.linnum)
        );
        call.setDoesNotReturn();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(AsmStatement *stmt) LLVM_OVERRIDE {
        AsmStatement_toIR(stmt, irs);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(CompoundAsmStatement *stmt) LLVM_OVERRIDE{
        CompoundAsmStatement_toIR(stmt, irs);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ImportStatement *stmt) LLVM_OVERRIDE {
        // Empty.
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(Statement *stmt) LLVM_OVERRIDE {
        error(stmt->loc, "Statement type Statement not implemented: %s", stmt->toChars());
        fatal();
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(PragmaStatement *stmt) LLVM_OVERRIDE {
        error(stmt->loc, "Statement type PragmaStatement not implemented: %s", stmt->toChars());
        fatal();
    }
};

//////////////////////////////////////////////////////////////////////////////

void Statement_toIR(Statement *s, IRState *irs)
{
    ToIRVisitor v(irs);
    s->accept(&v);
}
