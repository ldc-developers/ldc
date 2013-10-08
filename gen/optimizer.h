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

namespace llvm { class Module; }

bool ldc_optimize_module(llvm::Module* m, bool isCompleteExecutable);

// Returns whether the normal, full inlining pass will be run.
bool willInline();

bool optimize();

llvm::CodeGenOpt::Level codeGenOptLevel();

void verifyModule(llvm::Module* m);

#endif

