//===-- gen/optimizer.h - LLVM IR optimization ------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Handles the optimization of the generated LLVM modules according to the
// specified optimization level.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_OPTIMIZER_H
#define LDC_GEN_OPTIMIZER_H

// For llvm::CodeGenOpt::Level
#include "llvm/Support/CodeGen.h"

#include "llvm/Support/CommandLine.h"

namespace llvm {
class raw_ostream;
}

namespace opts {

enum SanitizerCheck {
  None,
  AddressSanitizer,
  MemorySanitizer,
  ThreadSanitizer
};

extern llvm::cl::opt<SanitizerCheck> sanitize;
}

namespace llvm {
class Module;
}

bool ldc_optimize_module(llvm::Module *m);

// Returns whether the normal, full inlining pass will be run.
bool willInline();

bool willCrossModuleInline();

bool isOptimizationEnabled();

llvm::CodeGenOpt::Level codeGenOptLevel();

void verifyModule(llvm::Module *m);

void outputOptimizationSettings(llvm::raw_ostream &hash_os);

#endif
