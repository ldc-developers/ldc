//===-- ir/irdsymbol.h - Codegen state for D vars/fields/params -*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Classes for representing the status of D variables on their way though the
// codegen process.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_IR_IRVAR_H
#define LDC_IR_IRVAR_H

#include "llvm/IR/Type.h"
#include "llvm/IR/DebugInfo.h"

struct IrFuncTyArg;
class VarDeclaration;

struct IrVar {
  explicit IrVar(VarDeclaration *var) : V(var) {}
  IrVar(VarDeclaration *var, llvm::Value *value) : V(var), value(value) {}

  VarDeclaration *V = nullptr;
  llvm::Value *value = nullptr;
};

// represents a global variable
struct IrGlobal : IrVar {
  explicit IrGlobal(VarDeclaration *v) : IrVar(v) {}
  IrGlobal(VarDeclaration *v, llvm::Type *type,
           llvm::Constant *constInit = nullptr)
      : IrVar(v), type(type), constInit(constInit) {}

  llvm::Type *type = nullptr;
  llvm::Constant *constInit = nullptr;

  // This var is used by a naked function.
  bool nakedUse = false;
};

// represents a local variable variable
struct IrLocal : IrVar {
  explicit IrLocal(VarDeclaration *v) : IrVar(v) {}
  IrLocal(VarDeclaration *v, llvm::Value *value) : IrVar(v, value) {}
  IrLocal(VarDeclaration *v, int nestedDepth, int nestedIndex)
      : IrVar(v), nestedDepth(nestedDepth), nestedIndex(nestedIndex) {}

  // Used for hybrid nested context creation.
  int nestedDepth = 0;
  int nestedIndex = -1;
};

// represents a function parameter
struct IrParameter : IrLocal {
  explicit IrParameter(VarDeclaration *v) : IrLocal(v) {}
  IrParameter(VarDeclaration *v, llvm::Value *value) : IrLocal(v, value) {}
  IrParameter(VarDeclaration *v, llvm::Value *value, IrFuncTyArg *arg,
              bool isVthis = false)
      : IrLocal(v, value), arg(arg), isVthis(isVthis) {}

  IrFuncTyArg *arg = nullptr;
  bool isVthis = false; // true, if it is the 'this' parameter
};

// represents an aggregate field variable
struct IrField : IrVar {
  explicit IrField(VarDeclaration *v) : IrVar(v){};
};

IrVar *getIrVar(VarDeclaration *decl);
llvm::Value *getIrValue(VarDeclaration *decl);
bool isIrVarCreated(VarDeclaration *decl);

IrGlobal *getIrGlobal(VarDeclaration *decl, bool create = false);
bool isIrGlobalCreated(VarDeclaration *decl);

IrLocal *getIrLocal(VarDeclaration *decl, bool create = false);
bool isIrLocalCreated(VarDeclaration *decl);

IrParameter *getIrParameter(VarDeclaration *decl, bool create = false);
bool isIrParameterCreated(VarDeclaration *decl);

IrField *getIrField(VarDeclaration *decl, bool create = false);
bool isIrFieldCreated(VarDeclaration *decl);

#endif
