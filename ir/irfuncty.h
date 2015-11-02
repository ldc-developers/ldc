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

#ifndef LDC_IR_IRFUNCTY_H
#define LDC_IR_IRFUNCTY_H

#include "llvm/ADT/SmallVector.h"

#include "gen/attributes.h"

#if defined(_MSC_VER)
#include "array.h"
#endif

#include <vector>

class DValue;
class Type;
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
  size_t parametersIdx = 0;

  /// This is the final LLVM Type used for the parameter/return value type
  llvm::Type *ltype = nullptr;

  /** These are the final LLVM attributes used for the function.
   *  Must be valid for the LLVM Type and byref setting */
  AttrBuilder attrs;

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
  IrFuncTyArg(Type *t, bool byref, AttrBuilder attrs = AttrBuilder());
};

// represents a function type
struct IrFuncTy {
  // The final LLVM type
  llvm::FunctionType *funcType = nullptr;

  // return value
  IrFuncTyArg *ret = nullptr;

  // null if not applicable
  IrFuncTyArg *arg_sret = nullptr;
  IrFuncTyArg *arg_this = nullptr;
  IrFuncTyArg *arg_nest = nullptr;
  IrFuncTyArg *arg_arguments = nullptr;

  // normal explicit arguments
  //    typedef llvm::SmallVector<IrFuncTyArg*, 4> ArgList;
  using ArgList = std::vector<IrFuncTyArg *>;
  ArgList args;

  // C varargs
  bool c_vararg = false;

  // range of normal parameters to reverse
  bool reverseParams = false;

  // reserved for ABI-specific data
  void *tag = nullptr;

  llvm::Value *putRet(DValue *dval);
  llvm::Value *getRet(Type *dty, llvm::Value *val);
  void getRet(Type *dty, llvm::Value *val, llvm::Value *address);

  llvm::Value *putParam(size_t idx, DValue *dval);
  llvm::Value *putParam(const IrFuncTyArg &arg, DValue *dval);
  void getParam(Type *dty, size_t idx, llvm::Value *val, llvm::Value *address);
};

#endif
