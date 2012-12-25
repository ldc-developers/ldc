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

#ifndef __LDC_GEN_ABI_H__
#define __LDC_GEN_ABI_H__

#include <vector>

struct Type;
struct TypeFunction;
struct IrFuncTyArg;
struct DValue;

namespace llvm
{
    class Type;
    class Value;
}

// return rewrite rule
struct ABIRewrite
{
    /// get a rewritten value back to its original form
    virtual llvm::Value* get(Type* dty, DValue* v) = 0;

    /// get a rewritten value back to its original form and store result in provided lvalue
    /// this one is optional and defaults to calling the one above
    virtual void getL(Type* dty, DValue* v, llvm::Value* lval);

    /// put out rewritten value
    virtual llvm::Value* put(Type* dty, DValue* v) = 0;

    /// should return the transformed type for this rewrite
    virtual llvm::Type* type(Type* dty, llvm::Type* t) = 0;
};

// interface called by codegen
struct TargetABI
{
    /// Returns the ABI for the target we're compiling for
    static TargetABI* getTarget();
    /// Returns the ABI for intrinsics
    static TargetABI* getIntrinsic();

    virtual void newFunctionType(TypeFunction* tf) {}
    virtual bool returnInArg(TypeFunction* tf) = 0;
    virtual bool passByVal(Type* t) = 0;
    virtual void doneWithFunctionType() {}

    virtual void rewriteFunctionType(TypeFunction* t) = 0;
};

#endif
