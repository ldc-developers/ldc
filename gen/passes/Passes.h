#ifndef LDC_PASSES_H
#define LDC_PASSES_H

#include "gen/metadata.h"
namespace llvm {
    class FunctionPass;
    class ModulePass;
}

// Performs simplifications on runtime calls.
llvm::FunctionPass* createSimplifyDRuntimeCalls();

#ifdef USE_METADATA
llvm::FunctionPass* createGarbageCollect2Stack();
llvm::ModulePass* createStripMetaData();
#endif

#if LLVM_REV >= 68940
llvm::ModulePass* createStripExternalsPass();
#endif

#endif
