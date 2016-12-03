//===-- ir/irfunction.h - Codegen state for D functions ---------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Represents the state of a D function/method/... on its way through the
// codegen process.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_IR_IRFUNCTION_H
#define LDC_IR_IRFUNCTION_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "gen/llvm.h"
#include "ir/irfuncty.h"
#include <stack>

class FuncDeclaration;
class TypeFunction;
class VarDeclaration;

// represents a function
struct IrFunction {
  // constructor
  explicit IrFunction(FuncDeclaration *fd);

  // annotations
  void setNeverInline();
  void setAlwaysInline();

  void setLLVMFunc(llvm::Function *function);

  /// Get 'real' llvm function, this function should be used if your need to
  /// access to function IR or attributes
  llvm::Function *getLLVMFunc() const;
  llvm::CallingConv::ID getCallingConv() const;
  llvm::FunctionType *getLLVMFuncType() const;

#if LDC_LLVM_VER >= 307
  bool hasLLVMPersonalityFn() const;
  void setLLVMPersonalityFn(llvm::Constant *personality);
#endif

  llvm::StringRef getLLVMFuncName() const;

  /// Get llvm function suitable for a calling, this function should be used
  /// if you need to make a call or to take function address
  llvm::Function *getCallee() const;

  FuncDeclaration *decl = nullptr;
  TypeFunction *type = nullptr;

  llvm::Value *sretArg = nullptr; // sret pointer arg
  llvm::Value *thisArg = nullptr; // class/struct 'this' arg
  llvm::Value *nestArg = nullptr; // nested function 'this' arg

  llvm::StructType *frameType = nullptr; // type of nested context
  unsigned frameTypeAlignment = 0;       // its alignment
  // number of enclosing functions with variables accessed by nested functions
  // (-1 if neither this function nor any enclosing ones access variables from
  // enclosing functions)
  int depth = -1;
  bool nestedContextCreated = false; // holds whether nested context is created

  // TODO: Move to FuncGenState?
  llvm::Value *_arguments = nullptr;
  llvm::Value *_argptr = nullptr;

#if LDC_LLVM_VER >= 307
  llvm::DISubprogram *diSubprogram = nullptr;
  std::stack<llvm::DILexicalBlock *> diLexicalBlocks;
  using VariableMap = llvm::DenseMap<VarDeclaration *, llvm::DILocalVariable *>;
#else
  llvm::DISubprogram diSubprogram;
  std::stack<llvm::DILexicalBlock> diLexicalBlocks;
  using VariableMap = llvm::DenseMap<VarDeclaration *, llvm::DIVariable>;
#endif
  // Debug info for all variables
  VariableMap variableMap;

  IrFuncTy irFty;

  /// Stores the FastMath options for this functions.
  /// These are set e.g. by math related UDA's from ldc.attributes.
  llvm::FastMathFlags FMF;

private:
  llvm::Function *func = nullptr;
};

IrFunction *getIrFunc(FuncDeclaration *decl, bool create = false);
bool isIrFuncCreated(FuncDeclaration *decl);

/// Get 'real' llvm function, this function should be used if your need to
/// access to function IR or attributes
llvm::Function *DtoFunc(FuncDeclaration *decl, bool create = false);

/// Get llvm function suitable for a calling, this function should be used
/// if you need to make a call or to take function address
llvm::Function *DtoCallee(FuncDeclaration *decl, bool create = false);

llvm::StringRef DtoFuncName(FuncDeclaration *decl);

#if LDC_LLVM_VER >= 308
void DtoSetFuncSubprogram(FuncDeclaration *decl, llvm::DISubprogram *SP);
#endif

#endif
