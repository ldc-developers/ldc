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
#if LDC_LLVM_VER < 1800
using CodeGenOptLevel = llvm::CodeGenOpt::Level;
#endif
}

namespace llvm {
class raw_ostream;
}

namespace llvm {
class Module;
class TargetLibraryInfoImpl;
}

bool ldc_optimize_module(llvm::Module *m);

// Returns whether the normal, full inlining pass will be run.
bool willInline();

bool willCrossModuleInline();

unsigned optLevel();

bool isOptimizationEnabled();

llvm::CodeGenOptLevel codeGenOptLevel();

void verifyModule(llvm::Module *m);

void outputOptimizationSettings(llvm::raw_ostream &hash_os);

std::unique_ptr<llvm::TargetLibraryInfoImpl> createTLII(llvm::Module &M);
