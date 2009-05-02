#ifndef LDC_PASSES_H
#define LDC_PASSES_H

#include "gen/metadata.h"
namespace llvm {
    class FunctionPass;
    class ModulePass;
}

// Performs simplifications on runtime calls.
llvm::FunctionPass* createSimplifyDRuntimeCalls();
llvm::FunctionPass* createGarbageCollect2Stack();

#ifdef USE_METADATA
llvm::ModulePass *createStripMetaData();
#endif


#endif
