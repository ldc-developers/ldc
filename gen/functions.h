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

#pragma once

class DValue;
class Expression;
class FuncDeclaration;
struct IRAsmBlock;
struct IrFuncTy;
struct Loc;
class Parameter;
class Type;
namespace llvm {
class FunctionType;
}

// Returns true if the function is a D/C main, eligible for implicit `return 0`
// (and void -> int return type fix-up).
bool isAnyMainFunction(FuncDeclaration *fd);

llvm::FunctionType *DtoFunctionType(Type *t, IrFuncTy &irFty, Type *thistype,
                                    Type *nesttype,
                                    FuncDeclaration *fd = nullptr);
llvm::FunctionType *DtoFunctionType(FuncDeclaration *fdecl);

void DtoResolveFunction(FuncDeclaration *fdecl);
void DtoDeclareFunction(FuncDeclaration *fdecl);
void DtoDefineFunction(FuncDeclaration *fd, bool linkageAvailableExternally = false);

void DtoDefineNakedFunction(FuncDeclaration *fd);
void emitABIReturnAsmStmt(IRAsmBlock *asmblock, const Loc &loc,
                          FuncDeclaration *fdecl);

DValue *DtoArgument(Parameter *fnarg, Expression *argexp);
