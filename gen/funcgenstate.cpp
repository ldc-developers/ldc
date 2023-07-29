//===-- funcgenstate.cpp --------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/funcgenstate.h"

#include "dmd/identifier.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/ms-cxx-helper.h"
#include "gen/runtime.h"
#include "ir/irfunction.h"

JumpTarget::JumpTarget(llvm::BasicBlock *targetBlock,
                       CleanupCursor cleanupScope, Statement *targetStatement)
    : targetBlock(targetBlock), cleanupScope(cleanupScope),
      targetStatement(targetStatement) {}

JumpTargets::JumpTargets(TryCatchFinallyScopes &scopes) : scopes(scopes) {}

void JumpTargets::pushLoopTarget(Statement *loopStatement,
                                 llvm::BasicBlock *continueTarget,
                                 llvm::BasicBlock *breakTarget) {
  continueTargets.emplace_back(continueTarget, scopes.currentCleanupScope(),
                               loopStatement);
  breakTargets.emplace_back(breakTarget, scopes.currentCleanupScope(),
                            loopStatement);
}

void JumpTargets::popLoopTarget() {
  continueTargets.pop_back();
  breakTargets.pop_back();
}

void JumpTargets::pushBreakTarget(Statement *switchStatement,
                                  llvm::BasicBlock *targetBlock) {
  breakTargets.push_back(
      {targetBlock, scopes.currentCleanupScope(), switchStatement});
}

void JumpTargets::popBreakTarget() { breakTargets.pop_back(); }

void JumpTargets::addLabelTarget(Identifier *labelName,
                                 llvm::BasicBlock *targetBlock) {
  labelTargets[labelName] = {targetBlock, scopes.currentCleanupScope(),
                             nullptr};

  // See whether any of the unresolved gotos target this label, and resolve
  // those that do.
  scopes.tryResolveGotos(labelName, targetBlock);
}

void JumpTargets::jumpToLabel(Loc loc, Identifier *labelName) {
  // If we have already seen that label, branch to it, executing any cleanups
  // as necessary.
  auto it = labelTargets.find(labelName);
  if (it != labelTargets.end()) {
    scopes.runCleanups(it->second.cleanupScope, it->second.targetBlock);
  } else {
    scopes.registerUnresolvedGoto(loc, labelName);
  }
}

void JumpTargets::jumpToStatement(std::vector<JumpTarget> &targets,
                                  Statement *loopOrSwitchStatement) {
  for (auto it = targets.rbegin(), end = targets.rend(); it != end; ++it) {
    if (it->targetStatement == loopOrSwitchStatement) {
      scopes.runCleanups(it->cleanupScope, it->targetBlock);
      return;
    }
  }
  assert(false && "Target for labeled break not found.");
}

void JumpTargets::jumpToClosest(std::vector<JumpTarget> &targets) {
  assert(!targets.empty() &&
         "Encountered break/continue but no loop in scope.");
  JumpTarget &t = targets.back();
  scopes.runCleanups(t.cleanupScope, t.targetBlock);
}

llvm::BasicBlock *SwitchCaseTargets::get(Statement *stmt) {
  auto it = targetBBs.find(stmt);
  assert(it != targetBBs.end());
  return it->second;
}

llvm::BasicBlock *SwitchCaseTargets::getOrCreate(Statement *stmt,
                                                 const llvm::Twine &name,
                                                 IRState &irs) {
  auto &bb = targetBBs[stmt];
  if (!bb)
    bb = irs.insertBB(name);
  return bb;
}

FuncGenState::FuncGenState(IrFunction &irFunc, IRState &irs)
    : irFunc(irFunc), scopes(irs), localVariableLifetimeAnnotator(irs),
      jumpTargets(scopes), switchTargets(), irs(irs) {}

llvm::CallBase *FuncGenState::callOrInvoke(llvm::Value *callee,
                                           llvm::FunctionType *calleeType,
                                           llvm::ArrayRef<llvm::Value *> args,
                                           const char *name, bool isNothrow) {
  // If this is a direct call, we might be able to use the callee attributes
  // to our advantage.
  llvm::Function *calleeFn = llvm::dyn_cast<llvm::Function>(callee);

  // Ignore 'nothrow' if there are active catch blocks handling non-Exception
  // Throwables.
  if (isNothrow && scopes.isCatchingNonExceptions())
    isNothrow = false;

  // Intrinsics don't support invoking and 'nounwind' functions don't need it.
  const bool doesNotThrow =
      isNothrow || global.params.betterC ||
      (calleeFn && (calleeFn->isIntrinsic() || calleeFn->doesNotThrow()));

  // calls inside a funclet must be annotated with its value
  llvm::SmallVector<llvm::OperandBundleDef, 2> BundleList;

  llvm::FunctionCallee calleeArg(calleeType, callee);

  if (doesNotThrow || scopes.empty()) {
    auto call = irs.ir->CreateCall(calleeArg, args, BundleList, name);
    if (calleeFn) {
      call->setAttributes(calleeFn->getAttributes());
    }
    return call;
  }

  llvm::BasicBlock *landingPad = scopes.getLandingPad();

  llvm::BasicBlock *postinvoke = irs.insertBB("postinvoke");
  auto invoke = irs.ir->CreateInvoke(calleeArg, postinvoke, landingPad, args,
                                     BundleList, name);
  if (calleeFn) {
    invoke->setAttributes(calleeFn->getAttributes());
  }

  irs.ir->SetInsertPoint(postinvoke);
  return invoke;
}
