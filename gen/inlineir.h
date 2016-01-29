//===-- gen/irstate.h - Inline IR implementation-----------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation for the LDC-specific LLVM inline IR feature.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_INLINEIR_H
#define LDC_GEN_INLINEIR_H

class FuncDeclaration;
namespace llvm {
class Function;
}

llvm::Function *DtoInlineIRFunction(FuncDeclaration *fdecl);

#endif
