//===-- gen/passes/Passes.h - LDC-specific LLVM passes ----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Functions for creating the LDC-specific LLVM optimizer passes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Config/llvm-config.h"
#include "llvm/IR/PassManager.h"

#if LLVM_VERSION_MAJOR >= 24
namespace llvm {
template <typename DerivedT>
using PassInfoMixin = detail::PassInfoMixin<DerivedT>;
}
#endif

namespace llvm {
class FunctionPass;
class ModulePass;
}

// Performs simplifications on runtime calls.
llvm::FunctionPass *createSimplifyDRuntimeCalls();

llvm::FunctionPass *createGarbageCollect2Stack();

llvm::FunctionPass *createWasmPointersSpillPass();

llvm::ModulePass *createStripExternalsPass();

llvm::ModulePass *createDLLImportRelocationPass();
