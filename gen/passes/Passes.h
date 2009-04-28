#ifndef LDC_PASSES_H
#define LDC_PASSES_H

namespace llvm {
    class FunctionPass;
}

// Performs simplifications on runtime calls.
llvm::FunctionPass* createSimplifyDRuntimeCalls();


#endif
