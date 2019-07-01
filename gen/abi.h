//===-- gen/abi.h - Target ABI description for IR generation ----*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// This interface is used by the IR generation code to accomodate any
// additional transformations necessary for the given target ABI (the direct
// LLVM IR representation for C structs unfortunately does not always lead to
// the right ABI, for example on x86_64).
//
//===----------------------------------------------------------------------===//

#pragma once

#include "dmd/mars.h"
#include "gen/dvalue.h"
#include "llvm/IR/CallingConv.h"
#include <vector>

class Type;
class TypeFunction;
class TypeStruct;
struct IrFuncTy;
struct IrFuncTyArg;
class FuncDeclaration;

namespace llvm {
class Type;
class Value;
class FunctionType;
}

/// Transforms function arguments and return values.
/// This is needed to implement certain ABI aspects which LLVM doesn't get
/// right by default.
struct ABIRewrite {
  virtual ~ABIRewrite() = default;

  /// Transforms the D argument to a suitable LL argument.
  virtual llvm::Value *put(DValue *v, bool isLValueExp, bool isLastArgExp) = 0;

  /// Transforms the LL parameter back and returns the address for the D
  /// parameter.
  virtual llvm::Value *getLVal(Type *dty, llvm::Value *v) = 0;

  /// Transforms the LL parameter back and returns the value for the D
  /// parameter.
  /// Defaults to loading the lvalue returned by getLVal().
  virtual llvm::Value *getRVal(Type *dty, llvm::Value *v);

  /// Returns the resulting LL type when transforming an argument of the
  /// specified D type.
  virtual llvm::Type *type(Type *t) = 0;

  /// Applies this rewrite to the specified argument, adapting it where
  /// necessary.
  virtual void applyTo(IrFuncTyArg &arg, llvm::Type *finalLType = nullptr);

protected:
  /***** Static Helpers *****/

  /// Returns the address of a D value, storing it to memory first if need be.
  static llvm::Value *getAddressOf(DValue *v);

  /// Loads a LL value of a specified type from memory. The element type of the
  /// provided pointer doesn't need to match the value type (=> suitable for
  /// bit-casting).
  static llvm::Value *loadFromMemory(llvm::Value *address, llvm::Type *asType,
                                     const char *name = ".bitcast_result");
};

// interface called by codegen
struct TargetABI {
  virtual ~TargetABI() = default;

  /// Returns the ABI for the target we're compiling for
  static TargetABI *getTarget();

  /// Returns the ABI for intrinsics
  static TargetABI *getIntrinsic();

  /// Returns the LLVM calling convention to be used for the given D linkage
  /// type on the target. Defaults to the C calling convention.
  virtual llvm::CallingConv::ID callingConv(LINK l, TypeFunction *tf = nullptr,
                                            FuncDeclaration *fdecl = nullptr) {
    return llvm::CallingConv::C;
  }

  /// Applies any rewrites that might be required to accurately reproduce the
  /// passed function name on LLVM given a specific calling convention.
  ///
  /// Using this function at a stage where the name could be user-visible is
  /// almost certainly a mistake; it is intended to e.g. prepend '\1' where
  /// disabling the LLVM-internal name mangling/postprocessing is required.
  virtual std::string mangleFunctionForLLVM(std::string name, LINK l) {
    return name;
  }

  /// Applies any rewrites that might be required to accurately reproduce the
  /// passed variable name on LLVM given a specific D linkage.
  ///
  /// Using this function at a stage where the name could be user-visible is
  /// almost certainly a mistake; it is intended to e.g. prepend '\1' where
  /// the LLVM-internal postprocessing of prepending a '_' must be disabled.
  virtual std::string mangleVariableForLLVM(std::string name, LINK l) {
    return name;
  }

  /// Returns true if all functions require the LLVM uwtable attribute.
  virtual bool needsUnwindTables() {
    // Condensed logic of Clang implementations of
    // `clang::ToolChain::IsUnwindTablesDefault()` based on early Clang 5.0.
    return global.params.targetTriple->getArch() == llvm::Triple::x86_64 ||
           global.params.targetTriple->getOS() == llvm::Triple::NetBSD;
  }

  /// Returns true if the D function uses sret (struct return).
  /// `needsThis` is true if the function type is for a non-static member
  /// function.
  ///
  /// A LL sret function doesn't really return a struct (in fact, it returns
  /// void); it merely just sets a struct which has been pre-allocated by the
  /// caller.
  /// The address is passed as additional function parameter using the StructRet
  /// attribute.
  virtual bool returnInArg(TypeFunction *tf, bool needsThis) = 0;

  /// Returns true if the D type is passed using the LLVM ByVal attribute.
  ///
  /// ByVal arguments are bitcopied to the callee's function parameters stack in
  /// memory.
  /// For the LL callee, a ByVal parameter is an implicit pointer to the
  /// bitcopy; the pointer is computed by LLVM and not passed as an explicit
  /// parameter.
  /// The LL caller needs to pass a pointer to the original argument (the memcpy
  /// source).
  virtual bool passByVal(TypeFunction *tf, Type *t) = 0;

  /// Returns true if the 'this' argument is to be passed before the 'sret'
  /// argument.
  virtual bool passThisBeforeSret(TypeFunction *tf) { return false; }

  /// Returns true if the explicit parameters order is to be reversed.
  /// Defaults to true for non-variadic extern(D) functions as required by
  /// druntime.
  virtual bool reverseExplicitParams(TypeFunction *tf);

  /// Called to give ABI the chance to rewrite the types
  virtual void rewriteFunctionType(IrFuncTy &fty) = 0;
  virtual void rewriteVarargs(IrFuncTy &fty, std::vector<IrFuncTyArg *> &args);
  virtual void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) {}

  /// Prepares a va_start intrinsic call by transforming the D argument (of type
  /// va_list) to a low-level value (of type i8*) to be passed to LLVM's
  /// va_start intrinsic.
  virtual llvm::Value *prepareVaStart(DLValue *ap);

  /// Implements the va_copy intrinsic.
  virtual void vaCopy(DLValue *dest, DValue *src);

  /// Prepares a va_arg intrinsic call by transforming the D argument (of type
  /// va_list) to a low-level value (of type i8*) to be passed to LLVM's
  /// va_arg intrinsic.
  virtual llvm::Value *prepareVaArg(DLValue *ap);

  /// Returns the D type to be used for va_list.
  ///
  /// Must match the alias in druntime.
  virtual Type *vaListType();

  /// Returns Objective-C message send function
  virtual const char *objcMsgSendFunc(Type *ret, IrFuncTy &fty);

  /***** Static Helpers *****/

  /// Check if struct 't' is a Homogeneous Floating-point Aggregate (HFA)
  /// consisting of up to 4 of same floating point type.  If so, optionally
  /// produce the rewriteType: an array of that floating point type
  static bool isHFA(TypeStruct *t, llvm::Type **rewriteType = nullptr, const int maxFloats = 4);

protected:

  /// Returns true if the D type is an aggregate:
  /// * struct
  /// * static/dynamic array
  /// * delegate
  /// * complex number
  static bool isAggregate(Type *t);

  /// Returns true if the D type is a Plain-Old-Datatype, optionally excluding
  /// structs with constructors from that definition.
  static bool isPOD(Type *t, bool excludeStructsWithCtor = false);

  /// Returns true if the D type can be bit-cast to an integer of the same size.
  static bool canRewriteAsInt(Type *t, bool include64bit = true);
};
