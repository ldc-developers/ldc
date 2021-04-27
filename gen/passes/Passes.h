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

namespace llvm {
class FunctionPass;
class ModulePass;
}

// Performs simplifications on runtime calls.
llvm::FunctionPass *createSimplifyDRuntimeCalls();

llvm::FunctionPass *createGarbageCollect2Stack();

llvm::ModulePass *createStripExternalsPass();

llvm::ModulePass *createDLLImportRelocationPass();
