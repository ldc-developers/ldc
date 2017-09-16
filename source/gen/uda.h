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
struct IrFunction;
namespace llvm {
class GlobalVariable;
}

void applyFuncDeclUDAs(FuncDeclaration *decl, IrFunction *irFunc);
void applyVarDeclUDAs(VarDeclaration *decl, llvm::GlobalVariable *gvar);

bool hasWeakUDA(Dsymbol *sym);
bool hasKernelAttr(Dsymbol *sym);
/// Must match ldc.dcompute.Compilefor + 1 == DComputeCompileFor
enum class DComputeCompileFor : int
{
  hostOnly = 0,
  deviceOnly = 1,
  hostAndDevice = 2
};
extern "C" DComputeCompileFor hasComputeAttr(Dsymbol *sym);
#endif
