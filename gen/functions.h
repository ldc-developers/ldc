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

struct DValue;
struct Expression;
struct FuncDeclaration;
struct IRAsmBlock;
struct Parameter;
struct Type;
namespace llvm
{
    class FunctionType;
    class Value;
}

llvm::FunctionType* DtoFunctionType(Type* t, Type* thistype, Type* nesttype, bool ismain = false);
llvm::FunctionType* DtoFunctionType(FuncDeclaration* fdecl);

llvm::FunctionType* DtoBaseFunctionType(FuncDeclaration* fdecl);

void DtoResolveFunction(FuncDeclaration* fdecl);
void DtoDeclareFunction(FuncDeclaration* fdecl);
void DtoDefineFunction(FuncDeclaration* fd);

void DtoDefineNakedFunction(FuncDeclaration* fd);
void emitABIReturnAsmStmt(IRAsmBlock* asmblock, Loc loc, FuncDeclaration* fdecl);

DValue* DtoArgument(Parameter* fnarg, Expression* argexp);
void DtoVariadicArgument(Expression* argexp, llvm::Value* dst);

#endif
