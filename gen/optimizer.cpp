#include "llvm/PassManager.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Target/TargetData.h"

using namespace llvm;

//////////////////////////////////////////////////////////////////////////////////////////

// this function runs some or all of the std-compile-opts passes depending on the
// optimization level given.

void llvmdc_optimize_module(Module* m, char lvl, bool doinline)
{
    assert(lvl >= 0 && lvl <= 5);
    if (lvl == 0)
        return;

    PassManager pm;
    pm.add(new TargetData(m));

    if (lvl >= 1)
    {
        pm.add(createRaiseAllocationsPass());
        pm.add(createCFGSimplificationPass());
        pm.add(createPromoteMemoryToRegisterPass());
    }

    if (lvl >= 2)
    {
        pm.add(createGlobalOptimizerPass());
        pm.add(createGlobalDCEPass());
        pm.add(createIPConstantPropagationPass());
        pm.add(createDeadArgEliminationPass());
        pm.add(createInstructionCombiningPass());
        pm.add(createCFGSimplificationPass());
        pm.add(createPruneEHPass());
    }

    if (doinline) {
        pm.add(createFunctionInliningPass());
    }

    if (lvl >= 3)
    {
        pm.add(createArgumentPromotionPass());
        pm.add(createTailDuplicationPass());
        pm.add(createInstructionCombiningPass());
        pm.add(createCFGSimplificationPass());
        pm.add(createScalarReplAggregatesPass());
        pm.add(createInstructionCombiningPass());
        pm.add(createCondPropagationPass());

        pm.add(createTailCallEliminationPass());
        pm.add(createCFGSimplificationPass());
        pm.add(createReassociatePass());
        pm.add(createLoopRotatePass());
        pm.add(createLICMPass());
        pm.add(createLoopUnswitchPass());
        pm.add(createInstructionCombiningPass());
        pm.add(createIndVarSimplifyPass());
        pm.add(createLoopUnrollPass());
        pm.add(createInstructionCombiningPass());
        pm.add(createGVNPass());
        pm.add(createSCCPPass());

        pm.add(createInstructionCombiningPass());
        pm.add(createCondPropagationPass());

        pm.add(createDeadStoreEliminationPass());
        pm.add(createAggressiveDCEPass());
        pm.add(createCFGSimplificationPass());
        pm.add(createSimplifyLibCallsPass());
        pm.add(createDeadTypeEliminationPass());
        pm.add(createConstantMergePass());
    }

    // level 4 and 5 are linktime optimizations

    pm.run(*m);
}
