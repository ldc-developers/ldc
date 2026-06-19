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

#pragma once

#include "llvm/IR/CallingConv.h"

class Dsymbol;
class FuncDeclaration;
class VarDeclaration;
class StructLiteralExp;
struct IrFunction;
namespace llvm {
class GlobalVariable;
}

void applyFuncDeclUDAs(FuncDeclaration *decl, IrFunction *irFunc);
void applyVarDeclUDAs(VarDeclaration *decl, llvm::GlobalVariable *gvar);

bool hasCallingConventionUDA(FuncDeclaration *fd, llvm::CallingConv::ID *callconv);
bool hasWeakUDA(Dsymbol *sym);
StructLiteralExp *getKernelAttr(Dsymbol *sym);
/// Must match ldc.dcompute.Compilefor + 1 == DComputeCompileFor
enum class DComputeCompileFor : int
{
  hostOnly = 0,
  deviceOnly = 1,
  hostAndDevice = 2
};
extern "C" DComputeCompileFor hasComputeAttr(Dsymbol *sym);

/// Returns whether `sym` is one of druntime's array comparison/equality
/// lowering hooks: `__cmp`/`__equals` and their helpers `isEqual`/`at` (from
/// core.internal.array.{comparison,equality}), plus `dstrcmp` (the char/string
/// `__cmp` helper from core.internal.string). Array `<`/`==` in `@compute`
/// device code lowers to these. They live in host-only modules but must still be
/// validated and codegen'd for the device, so they are exempted from the
/// host-only template skip.
bool isDeviceArrayComparisonHook(Dsymbol *sym);

bool hasNoSplitStackUDA(FuncDeclaration *fd);

unsigned getMaskFromNoSanitizeUDA(FuncDeclaration &fd);
