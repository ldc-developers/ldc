//===-- trycatchfinally.cpp -------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/trycatchfinally.h"

#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/statement.h"
#include "dmd/target.h"
#include "gen/classes.h"
#include "gen/funcgenstate.h"
#include "gen/llvmhelpers.h"
#include "gen/mangling.h"
#include "gen/ms-cxx-helper.h"
#include "gen/rttibuilder.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"
#include "ir/irtypeclass.h"

////////////////////////////////////////////////////////////////////////////////

TryCatchScope::TryCatchScope(IRState &irs, llvm::Value *ehPtrSlot,
                             TryCatchStatement *stmt, llvm::BasicBlock *endbb)
    : stmt(stmt), endbb(endbb) {
  assert(stmt->catches);

  cleanupScope = irs.funcGen().scopes.currentCleanupScope();
  catchesNonExceptions =
      std::any_of(stmt->catches->begin(), stmt->catches->end(), [](Catch *c) {
        for (auto cd = c->type->toBasetype()->isClassHandle(); cd;
             cd = cd->baseClass) {
          if (cd == ClassDeclaration::exception)
            return false;
        }
        return true;
      });

  if (useMSVCEH()) {
    emitCatchBodiesMSVC(irs, ehPtrSlot);
    return;
  }
  emitCatchBodies(irs, ehPtrSlot);
}

const std::vector<TryCatchScope::CatchBlock> &
TryCatchScope::getCatchBlocks() const {
  assert(!catchBlocks.empty());
  return catchBlocks;
}

void TryCatchScope::emitCatchBodies(IRState &irs, llvm::Value *ehPtrSlot) {
  assert(catchBlocks.empty());

  auto &PGO = irs.funcGen().pgo;
  const auto entryCount = PGO.setCurrentStmt(stmt);

  struct CBPrototype {
    ClassDeclaration *cd;
    llvm::BasicBlock *catchBB;
    uint64_t catchCount;
    uint64_t uncaughtCount;
  };
  llvm::SmallVector<CBPrototype, 8> cbPrototypes;
  cbPrototypes.reserve(stmt->catches->length);

  for (auto c : *stmt->catches) {
    auto catchBB =
        irs.insertBBBefore(endbb, llvm::Twine("catch.") + c->type->toChars());
    irs.ir->SetInsertPoint(catchBB);
    irs.DBuilder.EmitBlockStart(c->loc);
    PGO.emitCounterIncrement(c);

    const auto cd = c->type->toBasetype()->isClassHandle();
    const bool isCPPclass = cd->isCPPclass();

    const auto enterCatchFn = getRuntimeFunction(
        c->loc, irs.module,
        isCPPclass ? "__cxa_begin_catch" : "_d_eh_enter_catch");
    const auto ptr = DtoLoad(getVoidPtrType(), ehPtrSlot);
    const auto throwableObj = irs.ir->CreateCall(enterCatchFn, ptr);

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
    if (isCPPclass) {
      // from DMD:

      /* C++ catches need to end with call to __cxa_end_catch().
       * Create:
       *   try { handler } finally { __cxa_end_catch(); }
       * Note that this is worst case code because it always sets up an
       * exception handler. At some point should try to do better.
       */
      FuncDeclaration *fdend =
          FuncDeclaration::genCfunc(nullptr, Type::tvoid, "__cxa_end_catch");
      Expression *efunc = VarExp::create(Loc(), fdend);
      Expression *ecall = CallExp::create(Loc(), efunc);
      ecall->type = Type::tvoid;
      Statement *call = ExpStatement::create(Loc(), ecall);
      Statement *stmt =
          c->handler ? TryFinallyStatement::create(Loc(), c->handler, call)
                     : call;
      Statement_toIR(stmt, &irs);
    } else {
      if (c->handler)
        Statement_toIR(c->handler, &irs);
    }

    if (!irs.scopereturned())
      irs.ir->CreateBr(endbb);

    irs.DBuilder.EmitBlockEnd();

    // PGO information, currently unused
    auto catchCount = PGO.getRegionCount(c);
    // uncaughtCount is handled in a separate pass below

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

  catchBlocks.reserve(stmt->catches->length);

  for (const auto &p : cbPrototypes) {
    auto branchWeights =
        PGO.createProfileWeights(p.catchCount, p.uncaughtCount);

    DtoResolveClass(p.cd);

    LLGlobalVariable *ci;
    if (p.cd->isCPPclass()) {
      // Wrap std::type_info pointers inside a __cpp_type_info_ptr class
      // instance so that the personality routine may differentiate C++ catch
      // clauses from D ones.
      const auto wrapperMangle =
          getIRMangledAggregateName(p.cd, "18_cpp_type_info_ptr");

      ci = irs.module.getGlobalVariable(wrapperMangle);
      if (!ci) {
        const char *name = target.cpp.typeInfoMangle(p.cd);
        auto cpp_ti = declareGlobal(
            p.cd->loc, irs.module, getVoidPtrType(), name,
            /*isConstant*/ true, false, /*useDLLImport*/ p.cd->isExport());

        const auto cppTypeInfoPtrType = getCppTypeInfoPtrType();
        RTTIBuilder b(cppTypeInfoPtrType);
        b.push(cpp_ti);

        auto wrapperType = llvm::cast<llvm::StructType>(
            getIrType(cppTypeInfoPtrType)->isClass()->getMemoryLLType());
        auto wrapperInit = b.get_constant(wrapperType);

        ci = defineGlobal(p.cd->loc, irs.module, wrapperMangle, wrapperInit,
                          LLGlobalValue::LinkOnceODRLinkage,
                          /*isConstant=*/true);
      }
    } else {
      ci = getIrAggr(p.cd)->getClassInfoSymbol();
    }

    catchBlocks.push_back({ci, p.catchBB, branchWeights});
  }
}

namespace {
void emitBeginCatchMSVC(IRState &irs, Catch *ctch,
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
    const auto savedInsertPoint = irs.saveInsertPoint();
    irs.ir->SetInsertPoint(gIR->topallocapoint());
    DtoDeclarationExp(var);

    // catch handler will be outlined, so always treat as a nested reference
    exnObj = getIrValue(var);

    if (var->nestedrefs.length) {
      // if variable needed in a closure, use a stack temporary and copy it
      // when caught
      cpyObj = exnObj;
      exnObj = DtoAlloca(var->type, "exnObj");
    }
  } else if (ctch->type) {
    // catch without var
    exnObj = DtoAlloca(ctch->type, "exnObj");
  } else {
    // catch all
    exnObj = LLConstant::getNullValue(getVoidPtrType());
  }

  bool isCPPclass = false;
  if (ctch->type) {
    ClassDeclaration *cd = ctch->type->toBasetype()->isClassHandle();
    typeDesc = getTypeDescriptor(irs, cd);
    isCPPclass = cd->isCPPclass();
    if (!isCPPclass)
      clssInfo = getIrAggr(cd)->getClassInfoSymbol();
  } else {
    // catch all
    typeDesc = LLConstant::getNullValue(getVoidPtrType());
    clssInfo = LLConstant::getNullValue(DtoType(getClassInfoType()));
  }

  // "catchpad within %switch [TypeDescriptor, 0, &caughtObject]" must be
  // first instruction
  int flags = var ? (isCPPclass ? 8 : 0) : 64; // just mimicking clang here
  LLValue *args[] = {typeDesc, DtoConstUint(flags), exnObj};
  auto catchpad = irs.ir->CreateCatchPad(catchSwitchInst, args, "");
  catchSwitchInst->addHandler(irs.scopebb());

  if (cpyObj) {
    // assign the caught exception to the location in the closure
    auto val = irs.ir->CreateLoad(DtoType(var->type), exnObj);
    irs.ir->CreateStore(val, cpyObj);
    exnObj = cpyObj;
  }

  // Exceptions are never rethrown by D code (but thrown again), so
  // we can leave the catch handler right away and continue execution
  // outside the catch funclet
  llvm::BasicBlock *catchhandler = irs.insertBB("catchhandler");
  llvm::CatchReturnInst::Create(catchpad, catchhandler, irs.scopebb());
  irs.ir->SetInsertPoint(catchhandler);
  irs.funcGen().pgo.emitCounterIncrement(ctch);
  if (!isCPPclass) {
    auto enterCatchFn =
        getRuntimeFunction(ctch->loc, irs.module, "_d_eh_enter_catch");
    irs.CreateCallOrInvoke(enterCatchFn, exnObj, clssInfo);
  }
}
}

void TryCatchScope::emitCatchBodiesMSVC(IRState &irs, llvm::Value *) {
  assert(catchBlocks.empty());

  auto &scopes = irs.funcGen().scopes;

  auto catchSwitchBlock = irs.insertBBBefore(endbb, "catch.dispatch");
  llvm::BasicBlock *unwindto =
      scopes.currentCleanupScope() > 0 ? scopes.getLandingPad() : nullptr;
  auto catchSwitchInst = llvm::CatchSwitchInst::Create(
      llvm::ConstantTokenNone::get(irs.context()), unwindto,
      stmt->catches->length, "", catchSwitchBlock);

  for (auto c : *stmt->catches) {
    auto catchBB =
        irs.insertBBBefore(endbb, llvm::Twine("catch.") + c->type->toChars());

    irs.ir->SetInsertPoint(catchBB);
    irs.DBuilder.EmitBlockStart(c->loc);

    emitBeginCatchMSVC(irs, c, catchSwitchInst);

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
  if (!irs.func()->hasLLVMPersonalityFn()) {
    const char *personality = "__CxxFrameHandler3";
    irs.func()->setLLVMPersonalityFn(
        getRuntimeFunction(stmt->loc, irs.module, personality));
  }
}

////////////////////////////////////////////////////////////////////////////////

CleanupScope::CleanupScope(llvm::BasicBlock *beginBlock,
                           llvm::BasicBlock *endBlock) {
  if (useMSVCEH()) {
    findSuccessors(blocks, beginBlock, endBlock);
    return;
  }
  blocks.push_back(beginBlock);
  if (endBlock != beginBlock)
    blocks.push_back(endBlock);
}

llvm::BasicBlock *CleanupScope::run(IRState &irs, llvm::BasicBlock *sourceBlock,
                                    llvm::BasicBlock *continueWith) {
  if (useMSVCEH())
    return runCopying(irs, sourceBlock, continueWith);

  if (exitTargets.empty() || (exitTargets.size() == 1 &&
                              exitTargets[0].branchTarget == continueWith)) {
    // We didn't need a branch selector before and still don't need one.
    assert(!branchSelector);

    // Set up the unconditional branch at the end of the cleanup if we have
    // not done so already.
    if (exitTargets.empty()) {
      exitTargets.emplace_back(continueWith);
      llvm::BranchInst::Create(continueWith, endBlock());
    }
    exitTargets.front().sourceBlocks.push_back(sourceBlock);
    return beginBlock();
  }

  // We need a branch selector if we are here...
  if (!branchSelector) {
    // ... and have not created one yet, so do so now.
    llvm::Type *branchSelectorType = llvm::Type::getInt32Ty(irs.context());
    branchSelector = new llvm::AllocaInst(
        branchSelectorType, irs.module.getDataLayout().getAllocaAddrSpace(),
        llvm::Twine("branchsel.") + beginBlock()->getName(),
        irs.topallocapoint());

    // Now we also need to store 0 to it to keep the paths that go to the
    // only existing branch target the same.
    for (auto bb : exitTargets.front().sourceBlocks) {
      new llvm::StoreInst(DtoConstUint(0), branchSelector, bb->getTerminator());
    }

    // And convert the BranchInst to the existing branch target to a
    // SelectInst so we can append the other cases to it.
    endBlock()->getTerminator()->eraseFromParent();
    llvm::Value *sel =
        new llvm::LoadInst(branchSelectorType, branchSelector, "", endBlock());
    llvm::SwitchInst::Create(
        sel, exitTargets[0].branchTarget,
        1, // Expected number of branches, only for pre-allocating.
        endBlock());
  }

  // If we already know this branch target, figure out the branch selector
  // value and simply insert the store into the source block (prior to the
  // last instruction, which is the branch to the first cleanup).
  for (unsigned i = 0; i < exitTargets.size(); ++i) {
    CleanupExitTarget &t = exitTargets[i];
    if (t.branchTarget == continueWith) {
      new llvm::StoreInst(DtoConstUint(i), branchSelector,
                          sourceBlock->getTerminator());

      // Note: Strictly speaking, keeping this up to date would not be
      // needed right now, because we never to any optimizations that
      // require changes to the source blocks after the initial conversion
      // from one to two branch targets. Keeping this around for now to
      // ease future development, but may be removed to save some work.
      t.sourceBlocks.push_back(sourceBlock);

      return beginBlock();
    }
  }

  // We don't know this branch target yet, so add it to the SwitchInst...
  llvm::ConstantInt *const selectorVal = DtoConstUint(exitTargets.size());
  llvm::cast<llvm::SwitchInst>(endBlock()->getTerminator())
      ->addCase(selectorVal, continueWith);

  // ... insert the store into the source block...
  new llvm::StoreInst(selectorVal, branchSelector,
                      sourceBlock->getTerminator());

  // ... and keep track of it (again, this is unnecessary right now as
  // discussed in the above note).
  exitTargets.emplace_back(continueWith);
  exitTargets.back().sourceBlocks.push_back(sourceBlock);

  return beginBlock();
}

llvm::BasicBlock *CleanupScope::runCopying(IRState &irs,
                                           llvm::BasicBlock *sourceBlock,
                                           llvm::BasicBlock *continueWith,
                                           llvm::BasicBlock *unwindTo,
                                           llvm::Value *funclet) {
  if (isCatchSwitchBlock(beginBlock()))
    return continueWith;
  if (exitTargets.empty()) {
    if (!endBlock()->getTerminator())
      // Set up the unconditional branch at the end of the cleanup
      llvm::BranchInst::Create(continueWith, endBlock());
  } else {
    // check whether we have an exit target with the same continuation
    for (CleanupExitTarget &tgt : exitTargets)
      if (tgt.branchTarget == continueWith) {
        tgt.sourceBlocks.push_back(sourceBlock);
        return tgt.cleanupBlocks.front();
      }
  }

  // reuse the original IR if not unwinding and not already used
  bool useOriginal = unwindTo == nullptr && funclet == nullptr;
  for (CleanupExitTarget &tgt : exitTargets) {
    if (tgt.cleanupBlocks.front() == beginBlock()) {
      useOriginal = false;
      break;
    }
  }

  // append new target
  exitTargets.emplace_back(continueWith);
  auto &exitTarget = exitTargets.back();
  exitTarget.sourceBlocks.push_back(sourceBlock);

  if (useOriginal) {
    // change the continuation target if the initial branch was created
    // by another instance with unwinding
    if (continueWith)
      if (auto term = endBlock()->getTerminator())
        if (auto succ = term->getSuccessor(0))
          if (succ != continueWith)
            remapBlocksValue(blocks, succ, continueWith);
    exitTarget.cleanupBlocks = blocks;
  } else {
    // clone the code
    cloneBlocks(blocks, exitTarget.cleanupBlocks, continueWith, unwindTo,
                funclet);
  }
  return exitTarget.cleanupBlocks.front();
}

////////////////////////////////////////////////////////////////////////////////

TryCatchFinallyScopes::TryCatchFinallyScopes(IRState &irs) : irs(irs) {
  // create top-level stacks
  unresolvedGotosPerCleanupScope.emplace_back();
  landingPadsPerCleanupScope.emplace_back();
}

TryCatchFinallyScopes::~TryCatchFinallyScopes() {
  assert(currentCleanupScope() == 0);
  // If there are still unresolved gotos left, it means that they were either
  // down or "sideways" (i.e. down another branch) of the tree of all
  // cleanup scopes, both of which are not allowed in D.
  if (!currentUnresolvedGotos().empty()) {
    for (const auto &i : currentUnresolvedGotos()) {
      error(i.sourceLoc, "`goto` into `try`/`finally` scope is not allowed");
    }
    fatal();
  }
}

void TryCatchFinallyScopes::pushTryCatch(TryCatchStatement *stmt,
                                         llvm::BasicBlock *endbb) {
  TryCatchScope scope(irs, getOrCreateEhPtrSlot(), stmt, endbb);
  // Only after emitting all the catch bodies, register the catch scopes.
  // This is so that (re)throwing inside a catch does not match later
  // catches.
  tryCatchScopes.push_back(scope);

  if (!useMSVCEH())
    landingPadsPerCleanupScope[currentCleanupScope()].push_back(nullptr);
}

void TryCatchFinallyScopes::popTryCatch() {
  tryCatchScopes.pop_back();
  if (useMSVCEH()) {
    assert(isCatchSwitchBlock(cleanupScopes.back().beginBlock()));
    popCleanups(currentCleanupScope() - 1);
  } else {
    landingPadsPerCleanupScope[currentCleanupScope()].pop_back();
  }
}

bool TryCatchFinallyScopes::isCatchingNonExceptions() const {
  return std::any_of(
      tryCatchScopes.begin(), tryCatchScopes.end(),
      [](const TryCatchScope &tc) { return tc.isCatchingNonExceptions(); });
}

////////////////////////////////////////////////////////////////////////////////

void TryCatchFinallyScopes::pushCleanup(llvm::BasicBlock *beginBlock,
                                        llvm::BasicBlock *endBlock) {
  cleanupScopes.emplace_back(beginBlock, endBlock);
  unresolvedGotosPerCleanupScope.emplace_back();
  landingPadsPerCleanupScope.emplace_back();
}

void TryCatchFinallyScopes::popCleanups(CleanupCursor targetScope) {
  assert(targetScope <= currentCleanupScope());
  if (targetScope == currentCleanupScope())
    return;

  for (CleanupCursor i = currentCleanupScope(); i-- > targetScope;) {
    // Any gotos that are still unresolved necessarily leave this scope.
    // Thus, the cleanup needs to be executed.
    for (const auto &gotoJump : currentUnresolvedGotos()) {
      // Replace all branches to the tentative target by branches to the cleanup
      // and continue with the tentative target (we simply reuse it because
      // there is no reason not to).
      llvm::BasicBlock *tentative = gotoJump.tentativeTarget;
      // 1) Replace all branches to the tentative target by branches to a
      //    temporary placeholder BB.
      llvm::BasicBlock *dummy = irs.insertBB("");
      tentative->replaceAllUsesWith(dummy);
      // 2) We need a cleanup instance which continues execution with the
      //    tentative target.
      auto startCleanup =
          cleanupScopes[i].run(irs, gotoJump.sourceBlock, tentative);
      // 3) Replace all branches to the placeholder BB by branches to the
      //    cleanup.
      dummy->replaceAllUsesWith(startCleanup);
      dummy->eraseFromParent();
    }

    Gotos &nextUnresolved = unresolvedGotosPerCleanupScope[i];
    nextUnresolved.insert(nextUnresolved.end(),
                          currentUnresolvedGotos().begin(),
                          currentUnresolvedGotos().end());

    cleanupScopes.pop_back();
    unresolvedGotosPerCleanupScope.pop_back();
    landingPadsPerCleanupScope.pop_back();
  }
}

void TryCatchFinallyScopes::runCleanups(CleanupCursor targetScope,
                                        llvm::BasicBlock *continueWith) {
  runCleanups(currentCleanupScope(), targetScope, continueWith);
}

void TryCatchFinallyScopes::runCleanups(CleanupCursor sourceScope,
                                        CleanupCursor targetScope,
                                        llvm::BasicBlock *continueWith) {
  if (useMSVCEH()) {
    runCleanupCopies(sourceScope, targetScope, continueWith);
    return;
  }

  assert(targetScope <= sourceScope);

  if (targetScope == sourceScope) {
    // No cleanups to run, just branch to the next block.
    irs.ir->CreateBr(continueWith);
    return;
  }

  // Insert the unconditional branch to the first cleanup block.
  irs.ir->CreateBr(cleanupScopes[sourceScope - 1].beginBlock());

  // Update all the control flow in the cleanups to make sure we end up where
  // we want.
  for (CleanupCursor i = sourceScope; i-- > targetScope;) {
    llvm::BasicBlock *nextBlock =
        (i > targetScope) ? cleanupScopes[i - 1].beginBlock() : continueWith;
    cleanupScopes[i].run(irs, irs.scopebb(), nextBlock);
  }
}

void TryCatchFinallyScopes::runCleanupCopies(CleanupCursor sourceScope,
                                             CleanupCursor targetScope,
                                             llvm::BasicBlock *continueWith) {
  assert(targetScope <= sourceScope);

  // work through the blocks in reverse execution order, so we
  // can merge cleanups that end up at the same continuation target
  for (CleanupCursor i = targetScope; i < sourceScope; ++i)
    continueWith =
        cleanupScopes[i].runCopying(irs, irs.scopebb(), continueWith);

  // Insert the unconditional branch to the first cleanup block.
  irs.ir->CreateBr(continueWith);
}

////////////////////////////////////////////////////////////////////////////////

std::vector<GotoJump> &TryCatchFinallyScopes::currentUnresolvedGotos() {
  return unresolvedGotosPerCleanupScope[currentCleanupScope()];
}

void TryCatchFinallyScopes::registerUnresolvedGoto(Loc loc,
                                                   Identifier *labelName) {
  llvm::BasicBlock *target = irs.insertBB("goto.unresolved");
  irs.ir->CreateBr(target);
  currentUnresolvedGotos().push_back({loc, irs.scopebb(), target, labelName});
}

void TryCatchFinallyScopes::tryResolveGotos(Identifier *labelName,
                                            llvm::BasicBlock *targetBlock) {
  auto &unresolved = currentUnresolvedGotos();
  size_t i = 0;
  while (i < unresolved.size()) {
    if (unresolved[i].targetLabel != labelName) {
      ++i;
      continue;
    }

    unresolved[i].tentativeTarget->replaceAllUsesWith(targetBlock);
    unresolved[i].tentativeTarget->eraseFromParent();
    unresolved.erase(unresolved.begin() + i);
  }
}

////////////////////////////////////////////////////////////////////////////////

llvm::BasicBlock *TryCatchFinallyScopes::getLandingPad() {
  llvm::BasicBlock *&landingPad = getLandingPadRef(currentCleanupScope());
  if (!landingPad)
    landingPad = emitLandingPad();
  return landingPad;
}

llvm::BasicBlock *&
TryCatchFinallyScopes::getLandingPadRef(CleanupCursor scope) {
  auto &pads = landingPadsPerCleanupScope[scope];
  if (pads.empty()) {
    // Have not encountered any catches (for which we would push a scope) or
    // calls to throwing functions (where we would have already executed
    // this if) in this cleanup scope yet.
    pads.push_back(nullptr);
  }
  return pads.back();
}

namespace {
  llvm::LandingPadInst *createLandingPadInst(IRState &irs) {
    LLType *retType = LLStructType::get(getVoidPtrType(irs.context()),
                                        LLType::getInt32Ty(irs.context()));
    if (!irs.func()->hasLLVMPersonalityFn()) {
      irs.func()->setLLVMPersonalityFn(
          getRuntimeFunction(Loc(), irs.module, "_d_eh_personality"));
    }
  return irs.ir->CreateLandingPad(retType, 0);
}
}

llvm::BasicBlock *TryCatchFinallyScopes::emitLandingPad() {
  if (useMSVCEH()) {
    assert(currentCleanupScope() > 0);
    return emitLandingPadMSVC(currentCleanupScope() - 1);
  }

  // save and rewrite scope
  const auto savedInsertPoint = irs.saveInsertPoint();

  // insert landing pads at the end of the function, in emission order,
  // to improve human-readability of the IR
  llvm::BasicBlock *beginBB = irs.insertBBBefore(nullptr, "landingPad");
  irs.ir->SetInsertPoint(beginBB);

  llvm::LandingPadInst *landingPad = createLandingPadInst(irs);

  // Stash away the exception object pointer and selector value into their
  // stack slots.
  llvm::Value *ehPtr = DtoExtractValue(landingPad, 0);
  irs.ir->CreateStore(ehPtr, getOrCreateEhPtrSlot());

  llvm::Value *ehSelector = DtoExtractValue(landingPad, 1);
  const auto ehSelectorType = ehSelector->getType();
  if (!ehSelectorSlot)
    ehSelectorSlot = DtoRawAlloca(ehSelectorType, 0, "eh.selector");
  irs.ir->CreateStore(ehSelector, ehSelectorSlot);

  // Add landingpad clauses, emit finallys and 'if' chain to catch the
  // exception.
  CleanupCursor lastCleanup = currentCleanupScope();
  for (auto it = tryCatchScopes.rbegin(), end = tryCatchScopes.rend();
       it != end; ++it) {
    const auto &tryCatchScope = *it;

    // Insert any cleanups in between the previous (inner-more) try-catch scope
    // and this one.
    const auto newCleanup = tryCatchScope.getCleanupScope();
    assert(lastCleanup >= newCleanup);
    if (lastCleanup > newCleanup) {
      landingPad->setCleanup(true);
      llvm::BasicBlock *afterCleanupBB =
          irs.insertBB(beginBB->getName() + llvm::Twine(".after.cleanup"));
      runCleanups(lastCleanup, newCleanup, afterCleanupBB);
      irs.ir->SetInsertPoint(afterCleanupBB);
      lastCleanup = newCleanup;
    }

    for (const auto &cb : tryCatchScope.getCatchBlocks()) {
      // Add the ClassInfo reference to the landingpad instruction so it is
      // emitted to the EH tables.
      landingPad->addClause(cb.classInfoPtr);

      llvm::BasicBlock *mismatchBB =
          irs.insertBB(beginBB->getName() + llvm::Twine(".mismatch"));

      // "Call" llvm.eh.typeid.for, which gives us the eh selector value to
      // compare the landing pad selector value with.
      llvm::Value *ehTypeId = irs.ir->CreateCall(
          GET_INTRINSIC_DECL(eh_typeid_for), cb.classInfoPtr);

      // Compare the selector value from the unwinder against the expected
      // one and branch accordingly.
      irs.ir->CreateCondBr(
          irs.ir->CreateICmpEQ(
              irs.ir->CreateLoad(ehSelectorType, ehSelectorSlot), ehTypeId),
          cb.bodyBB, mismatchBB, cb.branchWeights);
      irs.ir->SetInsertPoint(mismatchBB);
    }
  }

  // No catch matched. Execute all finallys and resume unwinding.
  auto resumeUnwindBlock = getOrCreateResumeUnwindBlock();
  if (lastCleanup > 0) {
    landingPad->setCleanup(true);
    runCleanups(lastCleanup, 0, resumeUnwindBlock);
  } else if (!tryCatchScopes.empty()) {
    // Directly convert the last mismatch branch into a branch to the
    // unwind resume block.
    irs.scopebb()->replaceAllUsesWith(resumeUnwindBlock);
    irs.scopebb()->eraseFromParent();
  } else {
    irs.ir->CreateBr(resumeUnwindBlock);
  }

  return beginBB;
}

llvm::AllocaInst *TryCatchFinallyScopes::getOrCreateEhPtrSlot() {
  if (!ehPtrSlot)
    ehPtrSlot = DtoRawAlloca(getVoidPtrType(), 0, "eh.ptr");
  return ehPtrSlot;
}

llvm::BasicBlock *TryCatchFinallyScopes::getOrCreateResumeUnwindBlock() {
  if (!resumeUnwindBlock) {
    resumeUnwindBlock = irs.insertBB("eh.resume");

    llvm::BasicBlock *oldBB = irs.scopebb();
    irs.ir->SetInsertPoint(resumeUnwindBlock);

    llvm::Function *resumeFn = getUnwindResumeFunction(Loc(), irs.module);
    irs.ir->CreateCall(resumeFn, DtoLoad(getVoidPtrType(), getOrCreateEhPtrSlot()));
    irs.ir->CreateUnreachable();

    irs.ir->SetInsertPoint(oldBB);
  }
  return resumeUnwindBlock;
}

llvm::BasicBlock *
TryCatchFinallyScopes::emitLandingPadMSVC(CleanupCursor cleanupScope) {
  if (!irs.func()->hasLLVMPersonalityFn()) {
    const char *personality = "__CxxFrameHandler3";
    irs.func()->setLLVMPersonalityFn(
        getRuntimeFunction(Loc(), irs.module, personality));
  }

  if (cleanupScope == 0)
    return runCleanupPad(cleanupScope, nullptr);

  llvm::BasicBlock *&pad = getLandingPadRef(cleanupScope);
  if (!pad)
    pad = emitLandingPadMSVC(cleanupScope - 1);

  return runCleanupPad(cleanupScope, pad);
}

llvm::BasicBlock *
TryCatchFinallyScopes::runCleanupPad(CleanupCursor scope,
                                     llvm::BasicBlock *unwindTo) {
  // a catch switch never needs to be cloned and is an unwind target itself
  if (isCatchSwitchBlock(cleanupScopes[scope].beginBlock()))
    return cleanupScopes[scope].beginBlock();

  // each cleanup block is bracketed by a pair of cleanuppad/cleanupret
  // instructions, any unwinding should also just continue at the next
  // cleanup block, e.g.:
  //
  // cleanuppad:
  //   %0 = cleanuppad within %funclet[]
  //   %frame = nullptr
  //   if (!_d_enter_cleanup(%frame)) br label %cleanupret
  //                                  else br label %copy
  //
  // copy:
  //   invoke _dtor to %cleanupret unwind %unwindTo [ "funclet"(token %0) ]
  //
  // cleanupret:
  //   _d_leave_cleanup(%frame)
  //   cleanupret %0 unwind %unwindTo
  //
  llvm::BasicBlock *cleanupbb = irs.insertBB("cleanuppad");
  auto funcletToken = llvm::ConstantTokenNone::get(irs.context());
  auto cleanuppad =
      llvm::CleanupPadInst::Create(funcletToken, {}, "", cleanupbb);

  llvm::BasicBlock *cleanupret = irs.insertBBAfter(cleanupbb, "cleanupret");

  // preparation to allocate some space on the stack where _d_enter_cleanup
  //  can place an exception frame (but not done here)
  auto frame = getNullPtr(getVoidPtrType());

  const auto savedInsertPoint = irs.saveInsertPoint();

  auto endFn = getRuntimeFunction(Loc(), irs.module, "_d_leave_cleanup");
  irs.ir->SetInsertPoint(cleanupret);
  irs.DBuilder.EmitStopPoint(irs.func()->decl->loc);
  irs.ir->CreateCall(endFn, frame,
                     {llvm::OperandBundleDef("funclet", cleanuppad)}, "");
  llvm::CleanupReturnInst::Create(cleanuppad, unwindTo, cleanupret);

  auto copybb = cleanupScopes[scope].runCopying(irs, cleanupbb, cleanupret,
                                                unwindTo, cleanuppad);

  auto beginFn = getRuntimeFunction(Loc(), irs.module, "_d_enter_cleanup");
  irs.ir->SetInsertPoint(cleanupbb);
  irs.DBuilder.EmitStopPoint(irs.func()->decl->loc);
  auto exec = irs.ir->CreateCall(
      beginFn, frame, {llvm::OperandBundleDef("funclet", cleanuppad)}, "");
  llvm::BranchInst::Create(copybb, cleanupret, exec, cleanupbb);

  return cleanupbb;
}
