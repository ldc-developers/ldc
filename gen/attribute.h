//===-- gen/attribute.h - Handling of @ldc.attribute ------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Contains helpers for handling of @ldc.attribute.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_ATTRBUTE_H
#define LDC_GEN_ATTRBUTE_H

// From LLVM
namespace llvm {
class Function;
}

// From DMD
class FuncDeclaration;

void DtoFuncDeclarationAttribute(FuncDeclaration *fdecl, llvm::Function *func);

#endif