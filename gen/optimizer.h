#ifndef LDC_GEN_OPTIMIZER_H
#define LDC_GEN_OPTIMIZER_H

// For llvm::CodeGenOpt::Level
#if LDC_LLVM_VER == 300
#include "llvm/Target/TargetMachine.h"
#else
#include "llvm/Support/CodeGen.h"
#endif

namespace llvm { class Module; }

bool ldc_optimize_module(llvm::Module* m);

// Determines whether the inliner will run in the -O<N> list of passes
bool doInline();
// Determines whether the inliner will be run at all.
bool willInline();

int optLevel();

bool optimize();

llvm::CodeGenOpt::Level codeGenOptLevel();

void verifyModule(llvm::Module* m);

#endif

