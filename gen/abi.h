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

#ifndef LDC_GEN_ABI_H
#define LDC_GEN_ABI_H

#include "mars.h"
#if LDC_LLVM_VER >= 303
#include "llvm/IR/CallingConv.h"
#else
#include "llvm/CallingConv.h"
#endif
#include <vector>

class Type;
class TypeFunction;
struct IrFuncTy;
struct IrFuncTyArg;
class DValue;

namespace llvm
{
    class Type;
    class Value;
    class FunctionType;
}

// return rewrite rule
struct ABIRewrite
{
    virtual ~ABIRewrite() {}

    /// get a rewritten value back to its original form
    virtual llvm::Value* get(Type* dty, llvm::Value* v) = 0;

    /// get a rewritten value back to its original form and store result in provided lvalue
    /// this one is optional and defaults to calling the one above
    virtual void getL(Type* dty, llvm::Value* v, llvm::Value* lval);

    /// put out rewritten value
    virtual llvm::Value* put(DValue* v) = 0;

    /// should return the transformed type for this rewrite
    virtual llvm::Type* type(Type* dty, llvm::Type* t) = 0;

protected:
    /***** Static Helpers *****/

    // Returns the address of a D value, storing it to memory first if need be.
    static llvm::Value* getAddressOf(DValue* v);

    // Stores a LL value to a specified memory address. The element type of the provided
    // pointer doesn't need to match the value type (=> suited for bit-casting).
    static void storeToMemory(llvm::Value* rval, llvm::Value* address);

    // Loads a LL value of a specified type from memory. The element type of the provided
    // pointer doesn't need to match the value type (=> suited for bit-casting).
    static llvm::Value* loadFromMemory(llvm::Value* address, llvm::Type* asType,
        const char* name = ".bitcast_result");
};

// interface called by codegen
struct TargetABI
{
    virtual ~TargetABI() {}

    /// Returns the ABI for the target we're compiling for
    static TargetABI* getTarget();

    /// Returns the ABI for intrinsics
    static TargetABI* getIntrinsic();

    /// Returns the LLVM calling convention to be used for the given D linkage
    /// type on the target. Defaults to the C calling convention.
    virtual llvm::CallingConv::ID callingConv(llvm::FunctionType* ft, LINK l)
    {
        return llvm::CallingConv::C;
    }

    /// Applies any rewrites that might be required to accurately reproduce the
    /// passed function name on LLVM given a specific calling convention.
    ///
    /// Using this function at a stage where the name could be user-visible is
    /// almost certainly a mistake; it is intended to e.g. prepend '\1' where
    /// disabling the LLVM-internal name mangling/postprocessing is required.
    virtual std::string mangleForLLVM(llvm::StringRef name, LINK l) { return name; }

    /// Returns true if the function uses sret (struct return),
    /// meaning that it gets a hidden pointer to a struct which has been pre-
    /// allocated by the caller.
    virtual bool returnInArg(TypeFunction* tf) = 0;

    /// Returns true if the type is passed by value
    virtual bool passByVal(Type* t) = 0;

    // Returns true if the 'this' argument is to be passed before the 'sret' argument.
    virtual bool passThisBeforeSret(TypeFunction* tf) { return false; }

    /// Called to give ABI the chance to rewrite the types
    virtual void rewriteFunctionType(TypeFunction* t, IrFuncTy& fty) = 0;
    virtual void rewriteVarargs(IrFuncTy& fty, std::vector<IrFuncTyArg*>& args);
    virtual void rewriteArgument(IrFuncTy& fty, IrFuncTyArg& arg) {}

    // Prepares a va_start intrinsic call.
    // Input:  pointer to passed ap argument (va_list*)
    // Output: value to be passed to LLVM's va_start intrinsic (void*)
    virtual llvm::Value* prepareVaStart(llvm::Value* pAp);

    // Implements the va_copy intrinsic.
    // Input: pointer to dest argument (va_list*) and src argument (va_list)
    virtual void vaCopy(llvm::Value* pDest, llvm::Value* src);

    // Prepares a va_arg intrinsic call.
    // Input:  pointer to passed ap argument (va_list*)
    // Output: value to be passed to LLVM's va_arg intrinsic (void*)
    virtual llvm::Value* prepareVaArg(llvm::Value* pAp);

    /// Returns the D type to be used for va_list.
    ///
    /// Must match the alias in druntime.
    virtual Type* vaListType();
};

#endif
