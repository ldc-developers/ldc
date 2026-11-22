//===-- gen/metadata.h - LDC-specific LLVM metadata definitions -*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Defines the types of LLVM metadata used for D-specific optimization passes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/IR/Metadata.h"

// *** Metadata for TypeInfo instances ***
// A metadata node for a TypeInfo instance will be named TD_PREFIX ~ <Name of
// TypeInfo global>. The node contains a single operand, an arbitrary constant
// value of the LLVM type corresponding to the D type the TypeInfo is for.
#define TD_PREFIX "llvm.ldc.typeinfo."

inline std::string getMetadataName(const char *prefix,
                                   llvm::GlobalVariable *forGlobal) {
  llvm::StringRef globalName = forGlobal->getName();
  assert(!globalName.empty());
  return (prefix + (globalName[0] == '\1' ? globalName.substr(1) : globalName))
      .str();
}
