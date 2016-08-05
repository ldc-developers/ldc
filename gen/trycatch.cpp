//===-- trycatch.cpp --------------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/trycatch.h"

#include "statement.h"
#include "gen/classes.h"
#include "gen/funcgenstate.h"
#include "gen/llvmhelpers.h"
#include "gen/ms-cxx-helper.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"

////////////////////////////////////////////////////////////////////////////////

namespace {

#if LDC_LLVM_VER >= 308
void emitBeginCatchMSVC(IRState &irs, Catch *ctch, llvm::BasicBlock *endbb,
                        llvm::CatchSwitchInst *catchSwitchInst) {
  VarDeclaration *var = ctch->var;
  // The MSVC/x86 build uses C++ exception handling
  // This needs a series of catch pads to match the exception
  // and the catch handler must be terminated by a catch return instruction
  LLValue *exnObj = nullptr;
  LLValue *cpyObj = nullptr;
  LLValue *typeDesc = nullptr;
  LLValue *clssInfo = nullptr;
  if (var) {
    // alloca storage for the variable, it always needs a place on the stack
    // do not initialize, this will be done by the C++ exception handler
    var->_init = nullptr;

    // redirect scope to avoid the generation of debug info before the
    // catchpad
    IRScope save = irs.scope();
    irs.scope() = IRScope(gIR->topallocapoint()->getParent());
    irs.scope().builder.SetInsertPoint(gIR->topallocapoint());
    DtoDeclarationExp(var);

    // catch handler will be outlined, so always treat as a nested reference
    exnObj = getIrValue(var);

    if (var->nestedrefs.dim) {
      // if variable needed in a closure, use a stack temporary and copy it
      // when caught
      cpyObj = exnObj;
      exnObj = DtoAlloca(var->type, "exnObj");
    }
    irs.scope() = save;
    irs.DBuilder.EmitStopPoint(ctch->loc); // re-set debug loc after the
                                           // SetInsertPoint(allocaInst) call
  } else if (ctch->type) {
    // catch without var
    exnObj = DtoAlloca(ctch->type, "exnObj");
  } else {
    // catch all
    exnObj = LLConstant::getNullValue(getVoidPtrType());
  }

  if (ctch->type) {
    ClassDeclaration *cd = ctch->type->toBasetype()->isClassHandle();
    typeDesc = getTypeDescriptor(irs, cd);
    clssInfo = getIrAggr(cd)->getClassInfoSymbol();
  } else {
    // catch all
    typeDesc = LLConstant::getNullValue(getVoidPtrType());
    clssInfo = LLConstant::getNullValue(DtoType(Type::typeinfoclass->type));
  }

  // "catchpad within %switch [TypeDescriptor, 0, &caughtObject]" must be
  // first instruction
  int flags = var ? 0 : 64; // just mimicking clang here
  LLValue *args[] = {typeDesc, DtoConstUint(flags), exnObj};
  auto catchpad = irs.ir->CreateCatchPad(catchSwitchInst,
                                         llvm::ArrayRef<LLValue *>(args), "");
  catchSwitchInst->addHandler(irs.scopebb());

  if (cpyObj) {
    // assign the caught exception to the location in the closure
    auto val = irs.ir->CreateLoad(exnObj);
    irs.ir->CreateStore(val, cpyObj);
    exnObj = cpyObj;
  }

  // Exceptions are never rethrown by D code (but thrown again), so
  // we can leave the catch handler right away and continue execution
  // outside the catch funclet
  llvm::BasicBlock *catchhandler =
      llvm::BasicBlock::Create(irs.context(), "catchhandler", irs.topfunc());
  llvm::CatchReturnInst::Create(catchpad, catchhandler, irs.scopebb());
  irs.scope() = IRScope(catchhandler);
  auto enterCatchFn =
      getRuntimeFunction(Loc(), irs.module, "_d_eh_enter_catch");
  irs.CreateCallOrInvoke(enterCatchFn, DtoBitCast(exnObj, getVoidPtrType()),
                         clssInfo);
}
#endif

////////////////////////////////////////////////////////////////////////////////

llvm::LandingPadInst *createLandingPadInst(IRState &irs) {
  LLType *retType =
      LLStructType::get(LLType::getInt8PtrTy(irs.context()),
                        LLType::getInt32Ty(irs.context()), nullptr);
#if LDC_LLVM_VER >= 307
  LLFunction *currentFunction = irs.func()->func;
  if (!currentFunction->hasPersonalityFn()) {
    LLFunction *personalityFn =
        getRuntimeFunction(Loc(), irs.module, "_d_eh_personality");
    currentFunction->setPersonalityFn(personalityFn);
  }
  return irs.ir->CreateLandingPad(retType, 0);
#else
  LLFunction *personalityFn =
      getRuntimeFunction(Loc(), irs.module, "_d_eh_personality");
  return irs.ir->CreateLandingPad(retType, personalityFn, 0);
#endif
}
}

////////////////////////////////////////////////////////////////////////////////

TryCatchScope::TryCatchScope(TryCatchStatement *stmt, llvm::BasicBlock *endbb,
                             size_t cleanupScope)
    : stmt(stmt), endbb(endbb), cleanupScope(cleanupScope) {
  assert(stmt->catches);
  catchesNonExceptions =
      std::any_of(stmt->catches->begin(), stmt->catches->end(), [](Catch *c) {
        for (auto cd = c->type->toBasetype()->isClassHandle(); cd;
             cd = cd->baseClass) {
          if (cd == ClassDeclaration::exception)
            return false;
        }
        return true;
      });
}

const std::vector<TryCatchScope::CatchBlock> &
TryCatchScope::getCatchBlocks() const {
  assert(!catchBlocks.empty());
  return catchBlocks;
}

void TryCatchScope::emitCatchBodies(IRState &irs) {
  assert(catchBlocks.empty());

#if LDC_LLVM_VER >= 308
  if (useMSVCEH()) {
    emitCatchBodiesMSVC(irs);
    return;
  }
#endif

  auto &PGO = irs.funcGen().pgo;
  const auto entryCount = PGO.setCurrentStmt(stmt);

  struct CBPrototype {
    ClassDeclaration *cd;
    llvm::BasicBlock *catchBB;
    uint64_t catchCount;
    uint64_t uncaughtCount;
  };
  llvm::SmallVector<CBPrototype, 8> cbPrototypes;
  cbPrototypes.reserve(stmt->catches->dim);

  for (auto c : *stmt->catches) {
    auto catchBB = llvm::BasicBlock::Create(
        irs.context(), llvm::Twine("catch.") + c->type->toChars(),
        irs.topfunc(), endbb);

    irs.scope() = IRScope(catchBB);
    irs.DBuilder.EmitBlockStart(c->loc);
    PGO.emitCounterIncrement(c);

    const auto enterCatchFn =
        getRuntimeFunction(Loc(), irs.module, "_d_eh_enter_catch");
    auto ptr = DtoLoad(irs.funcGen().getOrCreateEhPtrSlot());
    auto throwableObj = irs.ir->CreateCall(enterCatchFn, ptr);

    // For catches that use the Throwable object, create storage for it.
    // We will set it in the code that branches from the landing pads
    // (there might be more than one) to catchBB.
    if (c->var) {
      // This will alloca if we haven't already and take care of nested refs
      // if there are any.
      DtoDeclarationExp(c->var);

      // Copy the exception reference over from the _d_eh_enter_catch return
      // value.
      DtoStore(DtoBitCast(throwableObj, DtoType(c->var->type)),
               getIrLocal(c->var)->value);
    }

    // Emit handler, if there is one. The handler is zero, for instance,
    // when building 'catch { debug foo(); }' in non-debug mode.
    if (c->handler)
      Statement_toIR(c->handler, &irs);

    if (!irs.scopereturned())
      irs.ir->CreateBr(endbb);

    irs.DBuilder.EmitBlockEnd();

    // PGO information, currently unused
    auto catchCount = PGO.getRegionCount(c);
    // uncaughtCount is handled in a separate pass below

    auto cd = c->type->toBasetype()->isClassHandle();
    cbPrototypes.push_back({cd, catchBB, catchCount, 0});
  }

  // Total number of uncaught exceptions is equal to the execution count at
  // the start of the try block minus the one after the continuation.
  // uncaughtCount keeps track of the exception type mismatch count while
  // iterating through the catch block prototypes in reversed order.
  auto uncaughtCount = entryCount - PGO.getRegionCount(stmt);
  for (auto it = cbPrototypes.rbegin(), end = cbPrototypes.rend(); it != end;
       ++it) {
    it->uncaughtCount = uncaughtCount;
    // Add this catch block's match count to the uncaughtCount, because these
    // failed to match the remaining (lexically preceding) catch blocks.
    uncaughtCount += it->catchCount;
  }

  catchBlocks.reserve(stmt->catches->dim);

  for (const auto &p : cbPrototypes) {
    auto branchWeights =
        PGO.createProfileWeights(p.catchCount, p.uncaughtCount);
    DtoResolveClass(p.cd);
    auto ci = getIrAggr(p.cd)->getClassInfoSymbol();
    catchBlocks.push_back({ci, p.catchBB, branchWeights});
  }
}

#if LDC_LLVM_VER >= 308
void TryCatchScope::emitCatchBodiesMSVC(IRState &irs) {
  auto &PGO = irs.funcGen().pgo;
  auto &scopes = irs.funcGen().scopes;

  auto catchSwitchBlock =
      llvm::BasicBlock::Create(irs.context(), "catch.dispatch", irs.topfunc());
  llvm::BasicBlock *unwindto =
      scopes.currentCleanupScope() > 0 ? scopes.getLandingPad() : nullptr;
  auto catchSwitchInst = llvm::CatchSwitchInst::Create(
      llvm::ConstantTokenNone::get(irs.context()), unwindto, stmt->catches->dim,
      "", catchSwitchBlock);

  for (auto c : *stmt->catches) {
    auto catchBB = llvm::BasicBlock::Create(
        irs.context(), llvm::Twine("catch.") + c->type->toChars(),
        irs.topfunc(), endbb);

    irs.scope() = IRScope(catchBB);
    irs.DBuilder.EmitBlockStart(c->loc);
    PGO.emitCounterIncrement(c);

    emitBeginCatchMSVC(irs, c, endbb, catchSwitchInst);

    // Emit handler, if there is one. The handler is zero, for instance,
    // when building 'catch { debug foo(); }' in non-debug mode.
    if (c->handler)
      Statement_toIR(c->handler, &irs);

    if (!irs.scopereturned())
      irs.ir->CreateBr(endbb);

    irs.DBuilder.EmitBlockEnd();
  }

  scopes.pushCleanup(catchSwitchBlock, catchSwitchBlock);

  // if no landing pad is created, the catch blocks are unused, but
  // the verifier complains if there are catchpads without personality
  // so we can just set it unconditionally
  if (!irs.func()->func->hasPersonalityFn()) {
    const char *personality = "__CxxFrameHandler3";
    LLFunction *personalityFn =
        getRuntimeFunction(Loc(), irs.module, personality);
    irs.func()->func->setPersonalityFn(personalityFn);
  }
}
#endif

////////////////////////////////////////////////////////////////////////////////

void TryCatchScopes::push(TryCatchStatement *stmt, llvm::BasicBlock *endbb) {
  TryCatchScope scope(stmt, endbb, irs.funcGen().scopes.currentCleanupScope());
  // Only after emitting all the catch bodies, register the catch scopes.
  // This is so that (re)throwing inside a catch does not match later
  // catches.
  scope.emitCatchBodies(irs);
  tryCatchScopes.push_back(scope);
}

void TryCatchScopes::pop() {
  tryCatchScopes.pop_back();
  if (useMSVCEH()) {
    auto &scopes = irs.funcGen().scopes;
    assert(isCatchSwitchBlock(scopes.cleanupScopes.back().beginBlock));
    scopes.popCleanups(scopes.currentCleanupScope() - 1);
  }
}

bool TryCatchScopes::isCatchingNonExceptions() const {
  return std::any_of(
      tryCatchScopes.begin(), tryCatchScopes.end(),
      [](const TryCatchScope &tc) { return tc.isCatchingNonExceptions(); });
}

llvm::BasicBlock *TryCatchScopes::emitLandingPad() {
  auto &scopes = irs.funcGen().scopes;

#if LDC_LLVM_VER >= 308
  if (useMSVCEH()) {
    auto currentCleanupScope = scopes.currentCleanupScope();
    assert(currentCleanupScope > 0);
    return emitLandingPadMSVC(currentCleanupScope - 1);
  }
#endif

  // save and rewrite scope
  IRScope savedIRScope = irs.scope();

  llvm::BasicBlock *beginBB =
      llvm::BasicBlock::Create(irs.context(), "landingPad", irs.topfunc());
  irs.scope() = IRScope(beginBB);

  llvm::LandingPadInst *landingPad = createLandingPadInst(irs);

  // Stash away the exception object pointer and selector value into their
  // stack slots.
  llvm::Value *ehPtr = DtoExtractValue(landingPad, 0);
  irs.ir->CreateStore(ehPtr, irs.funcGen().getOrCreateEhPtrSlot());

  llvm::Value *ehSelector = DtoExtractValue(landingPad, 1);
  if (!irs.funcGen().ehSelectorSlot) {
    irs.funcGen().ehSelectorSlot =
        DtoRawAlloca(ehSelector->getType(), 0, "eh.selector");
  }
  irs.ir->CreateStore(ehSelector, irs.funcGen().ehSelectorSlot);

  // Add landingpad clauses, emit finallys and 'if' chain to catch the
  // exception.
  size_t lastCleanup = scopes.currentCleanupScope();
  for (auto it = tryCatchScopes.rbegin(), end = tryCatchScopes.rend();
       it != end; ++it) {
    const auto &tryCatchScope = *it;

    // Insert any cleanups in between the previous (inner-more) try-catch scope
    // and this one.
    const auto newCleanup = tryCatchScope.getCleanupScope();
    assert(lastCleanup >= newCleanup);
    if (lastCleanup > newCleanup) {
      landingPad->setCleanup(true);
      llvm::BasicBlock *afterCleanupBB = llvm::BasicBlock::Create(
          irs.context(), beginBB->getName() + llvm::Twine(".after.cleanup"),
          irs.topfunc());
      scopes.runCleanups(lastCleanup, newCleanup, afterCleanupBB);
      irs.scope() = IRScope(afterCleanupBB);
      lastCleanup = newCleanup;
    }

    for (const auto &cb : tryCatchScope.getCatchBlocks()) {
      // Add the ClassInfo reference to the landingpad instruction so it is
      // emitted to the EH tables.
      landingPad->addClause(cb.classInfoPtr);

      llvm::BasicBlock *mismatchBB = llvm::BasicBlock::Create(
          irs.context(), beginBB->getName() + llvm::Twine(".mismatch"),
          irs.topfunc());

      // "Call" llvm.eh.typeid.for, which gives us the eh selector value to
      // compare the landing pad selector value with.
      llvm::Value *ehTypeId =
          irs.ir->CreateCall(GET_INTRINSIC_DECL(eh_typeid_for),
                             DtoBitCast(cb.classInfoPtr, getVoidPtrType()));

      // Compare the selector value from the unwinder against the expected
      // one and branch accordingly.
      irs.ir->CreateCondBr(
          irs.ir->CreateICmpEQ(irs.ir->CreateLoad(irs.funcGen().ehSelectorSlot),
                               ehTypeId),
          cb.bodyBB, mismatchBB, cb.branchWeights);
      irs.scope() = IRScope(mismatchBB);
    }
  }

  // No catch matched. Execute all finallys and resume unwinding.
  if (lastCleanup > 0) {
    landingPad->setCleanup(true);
    scopes.runCleanups(lastCleanup, 0,
                       irs.funcGen().getOrCreateResumeUnwindBlock());
  } else if (!tryCatchScopes.empty()) {
    // Directly convert the last mismatch branch into a branch to the
    // unwind resume block.
    irs.scopebb()->replaceAllUsesWith(
        irs.funcGen().getOrCreateResumeUnwindBlock());
    irs.scopebb()->eraseFromParent();
  } else {
    irs.ir->CreateBr(irs.funcGen().getOrCreateResumeUnwindBlock());
  }

  irs.scope() = savedIRScope;
  return beginBB;
}

#if LDC_LLVM_VER >= 308
llvm::BasicBlock *TryCatchScopes::emitLandingPadMSVC(size_t cleanupScope) {
  auto &scopes = irs.funcGen().scopes;

  LLFunction *currentFunction = irs.func()->func;
  if (!currentFunction->hasPersonalityFn()) {
    const char *personality = "__CxxFrameHandler3";
    LLFunction *personalityFn =
        getRuntimeFunction(Loc(), irs.module, personality);
    currentFunction->setPersonalityFn(personalityFn);
  }

  if (cleanupScope == 0)
    return scopes.runCleanupPad(cleanupScope, nullptr);

  llvm::BasicBlock *&pad = scopes.getLandingPadRef(cleanupScope - 1);
  if (!pad)
    pad = emitLandingPadMSVC(cleanupScope - 1);

  return scopes.runCleanupPad(cleanupScope, pad);
}
#endif
