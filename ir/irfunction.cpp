//===-- irfunction.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/irstate.h"
#include "gen/runtime.h"
#include "gen/tollvm.h"
#include "ir/irdsymbol.h"
#include "ir/irfunction.h"
#include <sstream>

namespace {
void executeCleanup(IRState* irs, CleanupScope& scope,
    llvm::BasicBlock* sourceBlock, llvm::BasicBlock* continueWith
) {
    if (scope.exitTargets.empty() || (
        scope.exitTargets.size() == 1 &&
        scope.exitTargets[0].branchTarget == continueWith
    )) {
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
            llvm::Type::getInt32Ty(gIR->context()),
            llvm::Twine("branchsel.") + scope.beginBlock->getName(),
            irs->topallocapoint()
        );

        // Now we also need to store 0 to it to keep the paths that go to the
        // only existing branch target the same.
        std::vector<llvm::BasicBlock*>& v = scope.exitTargets.front().sourceBlocks;
        for (std::vector<llvm::BasicBlock*>::iterator it = v.begin(), end = v.end();
            it != end; ++it
        ) {
            new llvm::StoreInst(DtoConstUint(0), scope.branchSelector,
                (*it)->getTerminator());
        }

        // And convert the BranchInst to the existing branch target to a
        // SelectInst so we can append the other cases to it.
        scope.endBlock->getTerminator()->eraseFromParent();
        llvm::Value* sel = new llvm::LoadInst(scope.branchSelector, "",
            scope.endBlock);
        llvm::SwitchInst::Create(
            sel,
            scope.exitTargets[0].branchTarget,
            1, // Expected number of branches, only for pre-allocating.
            scope.endBlock
        );
    }

    // If we already know this branch target, figure out the branch selector
    // value and simply insert the store into the source block (prior to the
    // last instruction, which is the branch to the first cleanup).
    for (unsigned i = 0; i < scope.exitTargets.size(); ++i) {
        CleanupExitTarget& t = scope.exitTargets[i];
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
    llvm::ConstantInt* const selectorVal = DtoConstUint(scope.exitTargets.size());
    llvm::cast<llvm::SwitchInst>(scope.endBlock->getTerminator())->addCase(
        selectorVal, continueWith);

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
        for (std::vector<GotoJump>::iterator it = topLevelUnresolvedGotos.begin(),
                                             end = topLevelUnresolvedGotos.end();
            it != end; ++it
        ) {
            error(it->sourceLoc, "goto into try/finally scope is not allowed");
        }
        fatal();
    }
}

void ScopeStack::pushCleanup(llvm::BasicBlock* beginBlock, llvm::BasicBlock* endBlock) {
    cleanupScopes.push_back(CleanupScope(beginBlock, endBlock));
}

void ScopeStack::runCleanups(
    CleanupCursor sourceScope,
    CleanupCursor targetScope,
    llvm::BasicBlock* continueWith
) {
    assert(targetScope <= sourceScope);

    if (targetScope == sourceScope) {
        // No cleanups to run, just branch to the next block.
        irs->ir->CreateBr(continueWith);
        return;
    }

    // Insert the unconditional branch to the first cleanup block.
    irs->ir->CreateBr(cleanupScopes[sourceScope - 1].beginBlock);

    // Update all the control flow in the cleanups to make sure we end up where
    // we want.
    for (CleanupCursor i = sourceScope; i-- > targetScope;) {
        llvm::BasicBlock* nextBlock = (i > targetScope) ?
            cleanupScopes[i - 1].beginBlock : continueWith;
        executeCleanup(irs, cleanupScopes[i], irs->scopebb(), nextBlock);
    }
}

void ScopeStack::runAllCleanups(llvm::BasicBlock* continueWith) {
    runCleanups(0, continueWith);
}

void ScopeStack::popCleanups(CleanupCursor targetScope) {
    assert(targetScope <= currentCleanupScope());
    if (targetScope == currentCleanupScope()) return;

    for (CleanupCursor i = currentCleanupScope(); i-- > targetScope;) {
        // Any gotos that are still unresolved necessarily leave this scope.
        // Thus, the cleanup needs to be executed.
        for (std::vector<GotoJump>::iterator it = currentUnresolvedGotos().begin(),
                                             end = currentUnresolvedGotos().end();
            it != end; ++it
        ) {
            // Make the source resp. last cleanup branch to this one.
            llvm::BasicBlock* tentative = it->tentativeTarget;
            tentative->replaceAllUsesWith(cleanupScopes[i].beginBlock);

            // And continue execution with the tentative target (we simply reuse
            // it because there is no reason not to).
            executeCleanup(irs, cleanupScopes[i], it->sourceBlock, tentative);
        }


        std::vector<GotoJump>& nextUnresolved = (i == 0) ?
            topLevelUnresolvedGotos : cleanupScopes[i - 1].unresolvedGotos;
        nextUnresolved.insert(
            nextUnresolved.end(),
            currentUnresolvedGotos().begin(),
            currentUnresolvedGotos().end()
        );

        cleanupScopes.pop_back();
    }
}

void ScopeStack::pushCatch(llvm::Constant* classInfoPtr,
    llvm::BasicBlock* bodyBlock
) {
    catchScopes.push_back({classInfoPtr, bodyBlock, currentCleanupScope()});
    currentLandingPads().push_back(0);
}

void ScopeStack::popCatch() {
    catchScopes.pop_back();
    currentLandingPads().pop_back();
}

void ScopeStack::pushLoopTarget(Statement* loopStatement, llvm::BasicBlock* continueTarget,
    llvm::BasicBlock* breakTarget
) {
    continueTargets.push_back({continueTarget, currentCleanupScope(), loopStatement});
    breakTargets.push_back({breakTarget, currentCleanupScope(), loopStatement});
}

void ScopeStack::popLoopTarget() {
    continueTargets.pop_back();
    breakTargets.pop_back();
}

void ScopeStack::pushBreakTarget(Statement* switchStatement,
    llvm::BasicBlock* targetBlock
) {
    breakTargets.push_back({targetBlock, currentCleanupScope(), switchStatement});
}

void ScopeStack::popBreakTarget() {
    breakTargets.pop_back();
}

void ScopeStack::addLabelTarget(Identifier* labelName,
    llvm::BasicBlock* targetBlock
) {
    labelTargets[labelName] = {targetBlock, currentCleanupScope(), 0};

    // See whether any of the unresolved gotos target this label, and resolve
    // those that do.
    std::vector<GotoJump>& unresolved = currentUnresolvedGotos();
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

void ScopeStack::jumpToLabel(Loc loc, Identifier* labelName) {
    // If we have already seen that label, branch to it, executing any cleanups
    // as necessary.
    LabelTargetMap::iterator it = labelTargets.find(labelName);
    if (it != labelTargets.end()) {
        runCleanups(it->second.cleanupScope, it->second.targetBlock);
        return;
    }

    llvm::BasicBlock* target =
        llvm::BasicBlock::Create(irs->context(), "goto.unresolved", irs->topfunc());
    irs->ir->CreateBr(target);
    currentUnresolvedGotos().push_back({loc, irs->scopebb(), target, labelName});
}

void ScopeStack::jumpToStatement(std::vector<JumpTarget>& targets,
    Statement* loopOrSwitchStatement
) {
    for (std::vector<JumpTarget>::reverse_iterator it = targets.rbegin(),
                                                   end = targets.rend();
        it != end; ++it
    ) {
        if (it->targetStatement == loopOrSwitchStatement) {
            runCleanups(it->cleanupScope, it->targetBlock);
            return;
        }
    }
    assert(false && "Target for labeled break not found.");
}

void ScopeStack::jumpToClosest(std::vector<JumpTarget>& targets) {
    assert(!targets.empty() &&
        "Encountered break/continue but no loop in scope.");
    JumpTarget &t = targets.back();
    runCleanups(t.cleanupScope, t.targetBlock);
}

std::vector<GotoJump>& ScopeStack::currentUnresolvedGotos() {
    return cleanupScopes.empty() ?
        topLevelUnresolvedGotos :
        cleanupScopes.back().unresolvedGotos;
}

std::vector<llvm::BasicBlock*>& ScopeStack::currentLandingPads() {
    return cleanupScopes.empty() ?
        topLevelLandingPads :
        cleanupScopes.back().landingPads;
}

namespace {
llvm::LandingPadInst* createLandingPadInst(IRState* irs) {
    LLType* retType = LLStructType::get(LLType::getInt8PtrTy(irs->context()),
                                        LLType::getInt32Ty(irs->context()),
                                        NULL);
#if LDC_LLVM_VER >= 307
    LLFunction* currentFunction = irs->func()->func;
    if (!currentFunction->hasPersonalityFn()) {
        LLFunction* personalityFn = LLVM_D_GetRuntimeFunction(Loc(), irs->module, "_d_eh_personality");
        currentFunction->setPersonalityFn(personalityFn);
    }
    return irs->ir->CreateLandingPad(retType, 0);
#else
    LLFunction* personalityFn = LLVM_D_GetRuntimeFunction(Loc(), irs->module, "_d_eh_personality");
    return irs->ir->CreateLandingPad(retType, personalityFn, 0);
#endif
}
}

llvm::BasicBlock* ScopeStack::emitLandingPad() {
    // save and rewrite scope
    IRScope savedIRScope = irs->scope();

    llvm::BasicBlock* beginBB = llvm::BasicBlock::Create(irs->context(),
        "landingPad", irs->topfunc());
    irs->scope() = IRScope(beginBB);

    llvm::LandingPadInst* landingPad = createLandingPadInst(irs);

    // Stash away the exception object pointer and selector value into their
    // stack slots.
    llvm::Value* ehPtr = DtoExtractValue(landingPad, 0);
    if (!irs->func()->resumeUnwindBlock) {
        irs->func()->resumeUnwindBlock = llvm::BasicBlock::Create(
            irs->context(), "unwind.resume", irs->topfunc());

        llvm::BasicBlock* oldBB = irs->scopebb();
        irs->scope() = IRScope(irs->func()->resumeUnwindBlock);

        llvm::Function* resumeFn = LLVM_D_GetRuntimeFunction(Loc(),
            irs->module, "_d_eh_resume_unwind");
        irs->ir->CreateCall(resumeFn,
            irs->ir->CreateLoad(irs->func()->getOrCreateEhPtrSlot()));
        irs->ir->CreateUnreachable();

        irs->scope() = IRScope(oldBB);
    }
    irs->ir->CreateStore(ehPtr, irs->func()->getOrCreateEhPtrSlot());

    llvm::Value* ehSelector = DtoExtractValue(landingPad, 1);
    if (!irs->func()->ehSelectorSlot) {
        irs->func()->ehSelectorSlot =
            DtoRawAlloca(ehSelector->getType(), 0, "eh.selector");
    }
    irs->ir->CreateStore(ehSelector, irs->func()->ehSelectorSlot);

    // Add landingpad clauses, emit finallys and 'if' chain to catch the exception.
    CleanupCursor lastCleanup = currentCleanupScope();
    for (std::vector<CatchScope>::reverse_iterator it = catchScopes.rbegin(),
                                                   end = catchScopes.rend();
        it != end; ++it
    ) {
        // Insert any cleanups in between the last catch we ran and this one.
        assert(lastCleanup >= it->cleanupScope);
        if (lastCleanup > it->cleanupScope) {
            landingPad->setCleanup(true);
            llvm::BasicBlock* afterCleanupBB = llvm::BasicBlock::Create(
                irs->context(),
                beginBB->getName() + llvm::Twine(".after.cleanup"),
                irs->topfunc()
            );
            runCleanups(lastCleanup, it->cleanupScope, afterCleanupBB);
            irs->scope() = IRScope(afterCleanupBB);
            lastCleanup = it->cleanupScope;
        }

        // Add the ClassInfo reference to the landingpad instruction so it is
        // emitted to the EH tables.
        landingPad->addClause(it->classInfoPtr);

        llvm::BasicBlock* mismatchBB = llvm::BasicBlock::Create(
            irs->context(),
            beginBB->getName() + llvm::Twine(".mismatch"),
            irs->topfunc()
        );

        // "Call" llvm.eh.typeid.for, which gives us the eh selector value to compare with
        llvm::Value* ehTypeId = irs->ir->CreateCall(GET_INTRINSIC_DECL(eh_typeid_for),
            DtoBitCast(it->classInfoPtr, getVoidPtrType()));

        // Compare the selector value from the unwinder against the expected
        // one and branch accordingly.
        irs->ir->CreateCondBr(
            irs->ir->CreateICmpEQ(
                irs->ir->CreateLoad(irs->func()->ehSelectorSlot), ehTypeId),
            it->bodyBlock,
            mismatchBB
        );
        irs->scope() = IRScope(mismatchBB);
    }

    // No catch matched. Execute all finallys and resume unwinding.
    if (lastCleanup > 0) {
        landingPad->setCleanup(true);
        runCleanups(lastCleanup, 0, irs->func()->resumeUnwindBlock);
    } else if (!catchScopes.empty()) {
        // Directly convert the last mismatch branch into a branch to the
        // unwind resume block.
        irs->scopebb()->replaceAllUsesWith(irs->func()->resumeUnwindBlock);
        irs->scopebb()->eraseFromParent();
    } else {
        irs->ir->CreateBr(irs->func()->resumeUnwindBlock);
    }

    irs->scope() = savedIRScope;
    return beginBB;
}

IrFunction::IrFunction(FuncDeclaration* fd) {
    decl = fd;

    Type* t = fd->type->toBasetype();
    assert(t->ty == Tfunction);
    type = static_cast<TypeFunction*>(t);
    func = NULL;
    allocapoint = NULL;

    retArg = NULL;
    thisArg = NULL;
    nestArg = NULL;

    nestedVar = NULL;
    frameType = NULL;
    frameTypeAlignment = 0;
    depth = -1;
    nestedContextCreated = false;

    _arguments = NULL;
    _argptr = NULL;

    retValSlot = NULL;
    retBlock = NULL;

    ehPtrSlot = NULL;
    resumeUnwindBlock = NULL;
    ehSelectorSlot = NULL;
}

void IrFunction::setNeverInline() {
#if LDC_LLVM_VER >= 303
    assert(!func->getAttributes().hasAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::AlwaysInline) && "function can't be never- and always-inline at the same time");
    func->addFnAttr(llvm::Attribute::NoInline);
#elif LDC_LLVM_VER == 302
    assert(!func->getFnAttributes().hasAttribute(llvm::Attributes::AlwaysInline) && "function can't be never- and always-inline at the same time");
    func->addFnAttr(llvm::Attributes::NoInline);
#else
    assert(!func->hasFnAttr(llvm::Attribute::AlwaysInline) && "function can't be never- and always-inline at the same time");
    func->addFnAttr(llvm::Attribute::NoInline);
#endif
}

void IrFunction::setAlwaysInline() {
#if LDC_LLVM_VER >= 303
    assert(!func->getAttributes().hasAttribute(llvm::AttributeSet::FunctionIndex, llvm::Attribute::NoInline) && "function can't be never- and always-inline at the same time");
    func->addFnAttr(llvm::Attribute::AlwaysInline);
#elif LDC_LLVM_VER == 302
    assert(!func->getFnAttributes().hasAttribute(llvm::Attributes::NoInline) && "function can't be never- and always-inline at the same time");
    func->addFnAttr(llvm::Attributes::AlwaysInline);
#else
    assert(!func->hasFnAttr(llvm::Attribute::NoInline) && "function can't be never- and always-inline at the same time");
    func->addFnAttr(llvm::Attribute::AlwaysInline);
#endif
}

llvm::AllocaInst* IrFunction::getOrCreateEhPtrSlot() {
    if (!ehPtrSlot) {
        ehPtrSlot = DtoRawAlloca(getVoidPtrType(), 0, "eh.ptr");
    }
    return ehPtrSlot;
}

IrFunction* getIrFunc(FuncDeclaration* decl, bool create)
{
    if (!isIrFuncCreated(decl) && create) {
        assert(decl->ir.irFunc == NULL);
        decl->ir.irFunc = new IrFunction(decl);
        decl->ir.m_type = IrDsymbol::FuncType;
    }
    assert(decl->ir.irFunc != NULL);
    return decl->ir.irFunc;
}

bool isIrFuncCreated(FuncDeclaration* decl) {
    int t = decl->ir.type();
    assert(t == IrDsymbol::FuncType || t == IrDsymbol::NotSet);
    return t == IrDsymbol::FuncType;
}
