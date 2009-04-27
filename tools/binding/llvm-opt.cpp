// Optimizer functionality for the LLVM D binding.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
#include "llvm/PassManager.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Target/TargetData.h"
#include "llvm-c/Core.h"

using namespace llvm;

extern "C" {

void LLVMOptimizeModule(LLVMModuleRef M, int doinline)
{
    Module* m = unwrap(M);

    PassManager pm;
    pm.add(new TargetData(m));

    //pm.add(createStripDeadPrototypesPass());
    pm.add(createGlobalDCEPass());

    pm.add(createRaiseAllocationsPass());
    pm.add(createCFGSimplificationPass());
    pm.add(createPromoteMemoryToRegisterPass());
    pm.add(createGlobalOptimizerPass());
    pm.add(createGlobalDCEPass());

    pm.add(createIPConstantPropagationPass());
    pm.add(createDeadArgEliminationPass());
    pm.add(createInstructionCombiningPass());
    pm.add(createCFGSimplificationPass());
    pm.add(createPruneEHPass());

    if (doinline)
        pm.add(createFunctionInliningPass());

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

    pm.run(*m);
}

}
