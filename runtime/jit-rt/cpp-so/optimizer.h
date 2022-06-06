//===-- optimizer.h - jit support -------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Jit runtime - jit optimizer.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

namespace llvm {
namespace legacy {
class PassManager;
class FunctionPassManager;
}
class TargetMachine;
class Module;
}

struct Context;

struct OptimizerSettings final {
  unsigned optLevel = 0;
  unsigned sizeLevel = 0;
};

void optimizeModule(const Context &context, llvm::TargetMachine &targetMachine,
                    const OptimizerSettings &settings, llvm::Module &module);

void setRtCompileVar(const Context &context, llvm::Module &module,
                     const char *name, const void *init);
