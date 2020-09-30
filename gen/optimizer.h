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

#pragma once

// For llvm::CodeGenOpt::Level
#include "llvm/Support/CodeGen.h"

#include "llvm/Support/CommandLine.h"

namespace llvm {
class raw_ostream;
}

namespace llvm {
class Module;
}

bool ldc_optimize_module(llvm::Module *m);

// Returns whether the normal, full inlining pass will be run.
bool willInline();

bool willCrossModuleInline();

#if LDC_LLVM_VER >= 800 && LDC_LLVM_VER < 1000
llvm::FramePointer::FP whichFramePointersToEmit();
#elif LDC_LLVM_VER < 800
bool willEliminateFramePointer();
#endif

unsigned optLevel();

bool isOptimizationEnabled();

llvm::CodeGenOpt::Level codeGenOptLevel();

void verifyModule(llvm::Module *m);

void outputOptimizationSettings(llvm::raw_ostream &hash_os);
