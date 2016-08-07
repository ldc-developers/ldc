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
  std::vector<GotoJump> &unresolved =
      tryCatchFinallyScopes.currentUnresolvedGotos();
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
  tryCatchFinallyScopes.currentUnresolvedGotos().emplace_back(
      loc, irs.scopebb(), target, labelName);
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
