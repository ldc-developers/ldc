//===-- gen/uda.h - Compiler-recognized UDA handling ------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// LDC supports "magic" UDAs in druntime (ldc.attribute) which are recognized
// by the compiler and influence code generation.
//
//===----------------------------------------------------------------------===//

#ifndef GEN_UDA_H
#define GEN_UDA_H

class Dsymbol;
class FuncDeclaration;
class VarDeclaration;
namespace llvm {
class Function;
class GlobalVariable;
}

void applyFuncDeclUDAs(FuncDeclaration *decl, llvm::Function *func);
void applyVarDeclUDAs(VarDeclaration *decl, llvm::GlobalVariable *gvar);

bool hasWeakUDA(Dsymbol *sym);

#endif
