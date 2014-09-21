//===-- gen/functions.h - D function codegen --------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Code generation for D function types and bodies.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_FUNCTIONS_H
#define LDC_GEN_FUNCTIONS_H

#include "mars.h"

class DValue;
class Expression;
class FuncDeclaration;
struct IRAsmBlock;
struct IrFuncTy;
class Parameter;
class Type;
namespace llvm
{
    class FunctionType;
    class Value;
}

llvm::FunctionType* DtoFunctionType(Type* t, IrFuncTy &irFty, Type* thistype, Type* nesttype,
                                    bool isMain = false, bool isCtor = false, bool isIntrinsic = false);
llvm::FunctionType* DtoFunctionType(FuncDeclaration* fdecl);

void DtoResolveFunction(FuncDeclaration* fdecl);
void DtoDeclareFunction(FuncDeclaration* fdecl);
void DtoDefineFunction(FuncDeclaration* fd);

void DtoDefineNakedFunction(FuncDeclaration* fd);
void emitABIReturnAsmStmt(IRAsmBlock* asmblock, Loc& loc, FuncDeclaration* fdecl);

DValue* DtoArgument(Parameter* fnarg, Expression* argexp);
void DtoVariadicArgument(Expression* argexp, llvm::Value* dst);

// Search for a druntime array op
int isDruntimeArrayOp(FuncDeclaration *fd);

#endif
