//===-- ir/irfuncty.h - Function type codegen metadata ----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Additional information attached to a function type during codegen. Handles
// LLVM attributes attached to a function and its parameters, etc.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "gen/attributes.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>

#if defined(_MSC_VER)
#include "dmd/root/array.h"
#endif

class DValue;
class Type;
class TypeFunction;
struct ABIRewrite;
namespace llvm {
class Type;
class Value;
class Instruction;
class Function;
class FunctionType;
}

/// Represents a function type argument (both explicit and implicit as well as
/// return values).
///
/// Instances of this only exist for arguments that are actually lowered to an
/// LLVM parameter (e.g. not for empty structs).
struct IrFuncTyArg {
  /** This is the original D type as the frontend knows it
   *  May NOT be rewritten!!! */
  Type *const type = nullptr;

  /// The index of the declaration in the FuncDeclaration::parameters array
  /// corresponding to this argument.
  size_t parametersIdx = -1;

  /// This is the final LLVM Type used for the parameter/return value type
  llvm::Type *ltype = nullptr;

  /** These are the final LLVM attributes used for the function.
   *  Must be valid for the LLVM Type and byref setting */
  llvm::AttrBuilder attrs;

  /** 'true' if the final LLVM argument is a LLVM reference type.
   *  Must be true when the D Type is a value type, but the final
   *  LLVM Type is a reference type! */
  bool byref = false;

  /** Pointer to the ABIRewrite structure needed to rewrite LLVM ValueS
   *  to match the final LLVM Type when passing arguments and getting
   *  return values */
  ABIRewrite *rewrite = nullptr;

  /// Helper to check if the 'inreg' attribute is set
  bool isInReg() const;
  /// Helper to check if the 'sret' attribute is set
  bool isSRet() const;
  /// Helper to check if the 'byval' attribute is set
  bool isByVal() const;

  /** @param t D type of argument/return value as known by the frontend
   *  @param byref Initial value for the 'byref' field. If true the initial
   *               LLVM Type will be of DtoType(type->pointerTo()), instead
   *               of just DtoType(type) */
  IrFuncTyArg(Type *t, bool byref);
  IrFuncTyArg(Type *t, bool byref, llvm::AttrBuilder);
  IrFuncTyArg(const IrFuncTyArg &) = delete;

  ~IrFuncTyArg();
};

// represents a function type
struct IrFuncTy {
  // D type
  TypeFunction *type;

  // The final LLVM type
  llvm::FunctionType *funcType = nullptr;

  // return value
  IrFuncTyArg *ret = nullptr;

  // null if not applicable
  IrFuncTyArg *arg_sret = nullptr;
  IrFuncTyArg *arg_this = nullptr;
  IrFuncTyArg *arg_nest = nullptr;
  IrFuncTyArg *arg_objcSelector = nullptr;
  IrFuncTyArg *arg_arguments = nullptr;

  // normal explicit arguments
  //    typedef llvm::SmallVector<IrFuncTyArg*, 4> ArgList;
  using ArgList = std::vector<IrFuncTyArg *>;
  ArgList args;

  // reserved for ABI-specific data
  void *tag = nullptr;

  llvm::Value *putRet(DValue *dval);
  llvm::Value *getRetRVal(Type *dty, llvm::Value *val);
  llvm::Value *getRetLVal(Type *dty, llvm::Value *val);

  llvm::Value *putArg(const IrFuncTyArg &arg, DValue *dval, bool isLValueExp,
                      bool isLastArgExp);
  llvm::Value *getParamLVal(Type *dty, size_t idx, llvm::Value *val);

  AttrSet getParamAttrs(bool passThisBeforeSret);

  IrFuncTy(TypeFunction *tf) : type(tf) {}
};
