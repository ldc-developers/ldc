//===-- funcgenstate.cpp --------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/funcgenstate.h"

#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/ms-cxx-helper.h"
#include "gen/runtime.h"
#include "ir/irfunction.h"

JumpTarget::JumpTarget(llvm::BasicBlock *targetBlock,
                       CleanupCursor cleanupScope, Statement *targetStatement)
    : targetBlock(targetBlock), cleanupScope(cleanupScope),
      targetStatement(targetStatement) {}

GotoJump::GotoJump(Loc loc, llvm::BasicBlock *sourceBlock,
                   llvm::BasicBlock *tentativeTarget, Identifier *targetLabel)
    : sourceLoc(std::move(loc)), sourceBlock(sourceBlock),
      tentativeTarget(tentativeTarget), targetLabel(targetLabel) {}

namespace {

#if LDC_LLVM_VER >= 308

// MSVC/x86 uses C++ exception handling that puts cleanup blocks into funclets.
// This means that we cannot use a branch selector and conditional branches
// at cleanup exit to continue with different targets.
// Instead we make a full copy of the cleanup code for every target
//
// Return the beginning basic block of the cleanup code
llvm::BasicBlock *executeCleanupCopying(IRState &irs, CleanupScope &scope,
                                        llvm::BasicBlock *sourceBlock,
                                        llvm::BasicBlock *continueWith,
                                        llvm::BasicBlock *unwindTo,
                                        llvm::Value *funclet) {
  if (isCatchSwitchBlock(scope.beginBlock))
    return continueWith;
  if (scope.cleanupBlocks.empty()) {
    // figure out the list of blocks used by this cleanup step
    findSuccessors(scope.cleanupBlocks, scope.beginBlock, scope.endBlock);
    if (!scope.endBlock->getTerminator())
      // Set up the unconditional branch at the end of the cleanup
      llvm::BranchInst::Create(continueWith, scope.endBlock);
  } else {
    // check whether we have an exit target with the same continuation
    for (CleanupExitTarget &tgt : scope.exitTargets)
      if (tgt.branchTarget == continueWith) {
        tgt.sourceBlocks.push_back(sourceBlock);
        return tgt.cleanupBlocks.front();
      }
  }

  // reuse the original IR if not unwinding and not already used
  bool useOriginal = unwindTo == nullptr && funclet == nullptr;
  for (CleanupExitTarget &tgt : scope.exitTargets)
    useOriginal = useOriginal && tgt.cleanupBlocks.front() != scope.beginBlock;

  // append new target
  scope.exitTargets.push_back(CleanupExitTarget(continueWith));
  scope.exitTargets.back().sourceBlocks.push_back(sourceBlock);

  if (useOriginal) {
    // change the continuation target if the initial branch was created
    // by another instance with unwinding
    if (continueWith)
      if (auto term = scope.endBlock->getTerminator())
        if (auto succ = term->getSuccessor(0))
          if (succ != continueWith) {
            remapBlocksValue(scope.cleanupBlocks, succ, continueWith);
          }
    scope.exitTargets.back().cleanupBlocks = scope.cleanupBlocks;
  } else {
    // clone the code
    cloneBlocks(scope.cleanupBlocks, scope.exitTargets.back().cleanupBlocks,
                continueWith, unwindTo, funclet);
  }
  return scope.exitTargets.back().cleanupBlocks.front();
}

#endif // LDC_LLVM_VER >= 308

void executeCleanup(IRState &irs, CleanupScope &scope,
                    llvm::BasicBlock *sourceBlock,
                    llvm::BasicBlock *continueWith) {
  assert(!useMSVCEH()); // should always use executeCleanupCopying

  if (scope.exitTargets.empty() ||
      (scope.exitTargets.size() == 1 &&
       scope.exitTargets[0].branchTarget == continueWith)) {
    // We didn't need a branch selector before and still don't need one.
    assert(!scope.branchSelector);

    // Set up the unconditional branch at the end of the cleanup if we have
    // not done so already.
    if (scope.exitTargets.empty()) {
      scope.exitTargets.push_back(CleanupExitTarget(continueWith));
      llvm::BranchInst::Create(continueWith, scope.endBlock);
    }
    scope.exitTargets.front().sourceBlocks.push_back(sourceBlock);
    return;
  }

  // We need a branch selector if we are here...
  if (!scope.branchSelector) {
    // ... and have not created one yet, so do so now.
    scope.branchSelector = new llvm::AllocaInst(
        llvm::Type::getInt32Ty(irs.context()),
        llvm::Twine("branchsel.") + scope.beginBlock->getName(),
        irs.topallocapoint());

    // Now we also need to store 0 to it to keep the paths that go to the
    // only existing branch target the same.
    auto &v = scope.exitTargets.front().sourceBlocks;
    for (auto bb : v) {
      new llvm::StoreInst(DtoConstUint(0), scope.branchSelector,
                          bb->getTerminator());
    }

    // And convert the BranchInst to the existing branch target to a
    // SelectInst so we can append the other cases to it.
    scope.endBlock->getTerminator()->eraseFromParent();
    llvm::Value *sel =
        new llvm::LoadInst(scope.branchSelector, "", scope.endBlock);
    llvm::SwitchInst::Create(
        sel, scope.exitTargets[0].branchTarget,
        1, // Expected number of branches, only for pre-allocating.
        scope.endBlock);
  }

  // If we already know this branch target, figure out the branch selector
  // value and simply insert the store into the source block (prior to the
  // last instruction, which is the branch to the first cleanup).
  for (unsigned i = 0; i < scope.exitTargets.size(); ++i) {
    CleanupExitTarget &t = scope.exitTargets[i];
    if (t.branchTarget == continueWith) {
      new llvm::StoreInst(DtoConstUint(i), scope.branchSelector,
                          sourceBlock->getTerminator());

      // Note: Strictly speaking, keeping this up to date would not be
      // needed right now, because we never to any optimizations that
      // require changes to the source blocks after the initial conversion
      // from one to two branch targets. Keeping this around for now to
      // ease future development, but may be removed to save some work.
      t.sourceBlocks.push_back(sourceBlock);

      return;
    }
  }

  // We don't know this branch target yet, so add it to the SwitchInst...
  llvm::ConstantInt *const selectorVal = DtoConstUint(scope.exitTargets.size());
  llvm::cast<llvm::SwitchInst>(scope.endBlock->getTerminator())
      ->addCase(selectorVal, continueWith);

  // ... insert the store into the source block...
  new llvm::StoreInst(selectorVal, scope.branchSelector,
                      sourceBlock->getTerminator());

  // ... and keep track of it (again, this is unnecessary right now as
  // discussed in the above note).
  scope.exitTargets.push_back(CleanupExitTarget(continueWith));
  scope.exitTargets.back().sourceBlocks.push_back(sourceBlock);
}
}

ScopeStack::~ScopeStack() {
  // If there are still unresolved gotos left, it means that they were either
  // down or "sideways" (i.e. down another branch) of the tree of all
  // cleanup scopes, both of which are not allowed in D.
  if (!topLevelUnresolvedGotos.empty()) {
    for (const auto &i : topLevelUnresolvedGotos) {
      error(i.sourceLoc, "goto into try/finally scope is not allowed");
    }
    fatal();
  }
}

void ScopeStack::pushCleanup(llvm::BasicBlock *beginBlock,
                             llvm::BasicBlock *endBlock) {
  cleanupScopes.push_back(CleanupScope(beginBlock, endBlock));
}

void ScopeStack::runCleanups(CleanupCursor sourceScope,
                             CleanupCursor targetScope,
                             llvm::BasicBlock *continueWith) {
#if LDC_LLVM_VER >= 308
  if (useMSVCEH()) {
    runCleanupCopies(sourceScope, targetScope, continueWith);
    return;
  }
#endif
  assert(targetScope <= sourceScope);

  if (targetScope == sourceScope) {
    // No cleanups to run, just branch to the next block.
    irs.ir->CreateBr(continueWith);
    return;
  }

  // Insert the unconditional branch to the first cleanup block.
  irs.ir->CreateBr(cleanupScopes[sourceScope - 1].beginBlock);

  // Update all the control flow in the cleanups to make sure we end up where
  // we want.
  for (CleanupCursor i = sourceScope; i-- > targetScope;) {
    llvm::BasicBlock *nextBlock =
        (i > targetScope) ? cleanupScopes[i - 1].beginBlock : continueWith;
    executeCleanup(irs, cleanupScopes[i], irs.scopebb(), nextBlock);
  }
}

#if LDC_LLVM_VER >= 308
void ScopeStack::runCleanupCopies(CleanupCursor sourceScope,
                                  CleanupCursor targetScope,
                                  llvm::BasicBlock *continueWith) {
  assert(targetScope <= sourceScope);

  // work through the blocks in reverse execution order, so we
  // can merge cleanups that end up at the same continuation target
  for (CleanupCursor i = targetScope; i < sourceScope; ++i)
    continueWith = executeCleanupCopying(irs, cleanupScopes[i], irs.scopebb(),
                                         continueWith, nullptr, nullptr);

  // Insert the unconditional branch to the first cleanup block.
  irs.ir->CreateBr(continueWith);
}

llvm::BasicBlock *ScopeStack::runCleanupPad(CleanupCursor scope,
                                            llvm::BasicBlock *unwindTo) {
  // a catch switch never needs to be cloned and is an unwind target itself
  if (isCatchSwitchBlock(cleanupScopes[scope].beginBlock))
    return cleanupScopes[scope].beginBlock;

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

  auto savedInsertBlock = irs.ir->GetInsertBlock();
  auto savedInsertPoint = irs.ir->GetInsertPoint();
  auto savedDbgLoc = irs.DBuilder.GetCurrentLoc();

  auto endFn = getRuntimeFunction(Loc(), irs.module, "_d_leave_cleanup");
  irs.ir->SetInsertPoint(cleanupret);
  irs.DBuilder.EmitStopPoint(irs.func()->decl->loc);
  irs.ir->CreateCall(endFn, frame,
                     {llvm::OperandBundleDef("funclet", cleanuppad)}, "");
  llvm::CleanupReturnInst::Create(cleanuppad, unwindTo, cleanupret);

  auto copybb = executeCleanupCopying(irs, cleanupScopes[scope], cleanupbb,
                                      cleanupret, unwindTo, cleanuppad);

  auto beginFn = getRuntimeFunction(Loc(), irs.module, "_d_enter_cleanup");
  irs.ir->SetInsertPoint(cleanupbb);
  irs.DBuilder.EmitStopPoint(irs.func()->decl->loc);
  auto exec = irs.ir->CreateCall(
      beginFn, frame, {llvm::OperandBundleDef("funclet", cleanuppad)}, "");
  llvm::BranchInst::Create(copybb, cleanupret, exec, cleanupbb);

  irs.ir->SetInsertPoint(savedInsertBlock, savedInsertPoint);
  irs.DBuilder.EmitStopPoint(savedDbgLoc);

  return cleanupbb;
}
#endif

void ScopeStack::runAllCleanups(llvm::BasicBlock *continueWith) {
  runCleanups(0, continueWith);
}

void ScopeStack::popCleanups(CleanupCursor targetScope) {
  assert(targetScope <= currentCleanupScope());
  if (targetScope == currentCleanupScope()) {
    return;
  }

  for (CleanupCursor i = currentCleanupScope(); i-- > targetScope;) {
    // Any gotos that are still unresolved necessarily leave this scope.
    // Thus, the cleanup needs to be executed.
    for (const auto &gotoJump : currentUnresolvedGotos()) {
      // Make the source resp. last cleanup branch to this one.
      llvm::BasicBlock *tentative = gotoJump.tentativeTarget;
#if LDC_LLVM_VER >= 308
      if (useMSVCEH()) {
        llvm::BasicBlock *afterCleanup = irs.insertBB("");
        auto startCleanup =
            executeCleanupCopying(irs, cleanupScopes[i], gotoJump.sourceBlock,
                                  afterCleanup, nullptr, nullptr);
        tentative->replaceAllUsesWith(startCleanup);
        afterCleanup->replaceAllUsesWith(tentative);
        afterCleanup->eraseFromParent();
      } else
#endif
      {
        tentative->replaceAllUsesWith(cleanupScopes[i].beginBlock);

        // And continue execution with the tentative target (we simply reuse
        // it because there is no reason not to).
        executeCleanup(irs, cleanupScopes[i], gotoJump.sourceBlock, tentative);
      }
    }

    std::vector<GotoJump> &nextUnresolved =
        (i == 0) ? topLevelUnresolvedGotos
                 : cleanupScopes[i - 1].unresolvedGotos;
    nextUnresolved.insert(nextUnresolved.end(),
                          currentUnresolvedGotos().begin(),
                          currentUnresolvedGotos().end());

    cleanupScopes.pop_back();
  }
}

void ScopeStack::pushTryCatch(TryCatchStatement *stmt,
                              llvm::BasicBlock *endbb) {
  tryCatchScopes.push(stmt, endbb);
  if (!useMSVCEH())
    currentLandingPads().push_back(nullptr);
}

void ScopeStack::popTryCatch() {
  tryCatchScopes.pop();
  if (!useMSVCEH())
    currentLandingPads().pop_back();
}

void ScopeStack::pushLoopTarget(Statement *loopStatement,
                                llvm::BasicBlock *continueTarget,
                                llvm::BasicBlock *breakTarget) {
  continueTargets.emplace_back(continueTarget, currentCleanupScope(),
                               loopStatement);
  breakTargets.emplace_back(breakTarget, currentCleanupScope(), loopStatement);
}

void ScopeStack::popLoopTarget() {
  continueTargets.pop_back();
  breakTargets.pop_back();
}

void ScopeStack::pushBreakTarget(Statement *switchStatement,
                                 llvm::BasicBlock *targetBlock) {
  breakTargets.push_back({targetBlock, currentCleanupScope(), switchStatement});
}

void ScopeStack::popBreakTarget() { breakTargets.pop_back(); }

void ScopeStack::addLabelTarget(Identifier *labelName,
                                llvm::BasicBlock *targetBlock) {
  labelTargets[labelName] = {targetBlock, currentCleanupScope(), nullptr};

  // See whether any of the unresolved gotos target this label, and resolve
  // those that do.
  std::vector<GotoJump> &unresolved = currentUnresolvedGotos();
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

void ScopeStack::jumpToLabel(Loc loc, Identifier *labelName) {
  // If we have already seen that label, branch to it, executing any cleanups
  // as necessary.
  auto it = labelTargets.find(labelName);
  if (it != labelTargets.end()) {
    runCleanups(it->second.cleanupScope, it->second.targetBlock);
    return;
  }

  llvm::BasicBlock *target = irs.insertBB("goto.unresolved");
  irs.ir->CreateBr(target);
  currentUnresolvedGotos().emplace_back(loc, irs.scopebb(), target, labelName);
}

void ScopeStack::jumpToStatement(std::vector<JumpTarget> &targets,
                                 Statement *loopOrSwitchStatement) {
  for (auto it = targets.rbegin(), end = targets.rend(); it != end; ++it) {
    if (it->targetStatement == loopOrSwitchStatement) {
      runCleanups(it->cleanupScope, it->targetBlock);
      return;
    }
  }
  assert(false && "Target for labeled break not found.");
}

void ScopeStack::jumpToClosest(std::vector<JumpTarget> &targets) {
  assert(!targets.empty() &&
         "Encountered break/continue but no loop in scope.");
  JumpTarget &t = targets.back();
  runCleanups(t.cleanupScope, t.targetBlock);
}

std::vector<GotoJump> &ScopeStack::currentUnresolvedGotos() {
  return cleanupScopes.empty() ? topLevelUnresolvedGotos
                               : cleanupScopes.back().unresolvedGotos;
}

std::vector<llvm::BasicBlock *> &ScopeStack::currentLandingPads() {
  return cleanupScopes.empty() ? topLevelLandingPads
                               : cleanupScopes.back().landingPads;
}

llvm::BasicBlock *&ScopeStack::getLandingPadRef(CleanupCursor scope) {
  auto &pads = cleanupScopes.empty() ? topLevelLandingPads
                                     : cleanupScopes[scope].landingPads;
  if (pads.empty()) {
    // Have not encountered any catches (for which we would push a scope) or
    // calls to throwing functions (where we would have already executed
    // this if) in this cleanup scope yet.
    pads.push_back(nullptr);
  }
  return pads.back();
}

llvm::BasicBlock *ScopeStack::getLandingPad() {
  llvm::BasicBlock *&landingPad = getLandingPadRef(currentCleanupScope() - 1);
  if (!landingPad)
    landingPad = tryCatchScopes.emitLandingPad();
  return landingPad;
}

llvm::BasicBlock *SwitchCaseTargets::get(Statement *stmt) {
  auto it = targetBBs.find(stmt);
  assert(it != targetBBs.end());
  return it->second;
}

llvm::BasicBlock *SwitchCaseTargets::getOrCreate(Statement *stmt,
                                                 const llvm::Twine &name) {
  auto &bb = targetBBs[stmt];
  if (!bb)
    bb = gIR->insertBB(name);
  return bb;
}

FuncGenState::FuncGenState(IrFunction &irFunc, IRState &irs)
    : irFunc(irFunc), scopes(irs), switchTargets(irFunc.func), irs(irs) {}

llvm::AllocaInst *FuncGenState::getOrCreateEhPtrSlot() {
  if (!ehPtrSlot) {
    ehPtrSlot = DtoRawAlloca(getVoidPtrType(), 0, "eh.ptr");
  }
  return ehPtrSlot;
}

llvm::BasicBlock *FuncGenState::getOrCreateResumeUnwindBlock() {
  assert(irFunc.func == irs.topfunc() &&
         "Should only access unwind resume block while emitting function.");
  if (!resumeUnwindBlock) {
    resumeUnwindBlock = irs.insertBB("eh.resume");

    llvm::BasicBlock *oldBB = irs.scopebb();
    irs.scope() = IRScope(resumeUnwindBlock);

    llvm::Function *resumeFn =
        getRuntimeFunction(Loc(), irs.module, "_d_eh_resume_unwind");
    irs.ir->CreateCall(resumeFn, DtoLoad(getOrCreateEhPtrSlot()));
    irs.ir->CreateUnreachable();

    irs.scope() = IRScope(oldBB);
  }
  return resumeUnwindBlock;
}
