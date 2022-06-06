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

#pragma once

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

  /// Returns the associated LLVM function.
  /// Use getLLVMCallee() for the LLVM function to be used for calls.
  llvm::Function *getLLVMFunc() const;
  llvm::CallingConv::ID getCallingConv() const;
  llvm::FunctionType *getLLVMFuncType() const;
  llvm::StringRef getLLVMFuncName() const;

  bool hasLLVMPersonalityFn() const;
  void setLLVMPersonalityFn(llvm::Constant *personality);

  /// Returns the associated LLVM function to be used for calls (potentially
  /// some sort of wrapper, e.g., a JIT wrapper).
  llvm::Function *getLLVMCallee() const;

  bool isDynamicCompiled() const;

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

  llvm::DISubprogram *diSubprogram = nullptr;
  std::stack<llvm::DILexicalBlock *> diLexicalBlocks;
  using VariableMap = llvm::DenseMap<VarDeclaration *, llvm::DILocalVariable *>;
  // Debug info for all variables
  VariableMap variableMap;

  IrFuncTy irFty;

  /// Stores the FastMath options for this functions.
  /// These are set e.g. by math related UDA's from ldc.attributes.
  llvm::FastMathFlags FMF;

  /// target CPU was overriden by attribute
  bool targetCpuOverridden = false;

  /// target features was overriden by attributes
  bool targetFeaturesOverridden = false;

  /// This functions was marked for dynamic compilation
  bool dynamicCompile = false;

  /// This functions was marked emit-only for dynamic compilation
  bool dynamicCompileEmit = false;

  /// Dynamic compilation thunk, all attempts to call or take address of the
  /// original function will be redirected to it
  llvm::Function *rtCompileFunc = nullptr;

private:
  llvm::Function *func = nullptr;
};

IrFunction *getIrFunc(FuncDeclaration *decl, bool create = false);
bool isIrFuncCreated(FuncDeclaration *decl);

/// Returns the associated LLVM function to be used for calls (potentially
/// some sort of wrapper, e.g., a JIT wrapper).
llvm::Function *DtoCallee(FuncDeclaration *decl, bool create = true);
