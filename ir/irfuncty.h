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

// represents a function type argument
// both explicit and implicit as well as return values
struct IrFuncTyArg
{
    /** This is the original D type as the frontend knows it
     *  May NOT be rewritten!!! */
    Type* const type;

    /// This is the final LLVM Type used for the parameter/return value type
    llvm::Type* ltype;

    /** These are the final LLVM attributes used for the function.
     *  Must be valid for the LLVM Type and byref setting */
    AttrBuilder attrs;

    /** 'true' if the final LLVM argument is a LLVM reference type.
     *  Must be true when the D Type is a value type, but the final
     *  LLVM Type is a reference type! */
    bool byref;

    /** Pointer to the ABIRewrite structure needed to rewrite LLVM ValueS
     *  to match the final LLVM Type when passing arguments and getting
     *  return values */
    ABIRewrite* rewrite;

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
    IrFuncTyArg(Type* t, bool byref, const AttrBuilder& attrs = AttrBuilder());
};

// represents a function type
struct IrFuncTy
{
    // The final LLVM type
    llvm::FunctionType* funcType;

    // return value
    IrFuncTyArg* ret;

    // null if not applicable
    IrFuncTyArg* arg_sret;
    IrFuncTyArg* arg_this;
    IrFuncTyArg* arg_nest;
    IrFuncTyArg* arg_arguments;

    // normal explicit arguments
//    typedef llvm::SmallVector<IrFuncTyArg*, 4> ArgList;
#if defined(_MSC_VER)
    typedef Array<IrFuncTyArg *> ArgList;
    typedef ArgList::iterator ArgIter;
    typedef ArgList::reverse_iterator ArgRIter;
#else
    typedef std::vector<IrFuncTyArg*> ArgList;
    typedef ArgList::iterator ArgIter;
    typedef ArgList::reverse_iterator ArgRIter;
#endif
    ArgList args;

    // C varargs
    bool c_vararg;

    // range of normal parameters to reverse
    bool reverseParams;

    // reserved for ABI-specific data
    void* tag;

    IrFuncTy()
    :   funcType(0),
        ret(NULL),
        arg_sret(NULL),
        arg_this(NULL),
        arg_nest(NULL),
        arg_arguments(NULL),
        args(),
        c_vararg(false),
        reverseParams(false),
        tag(NULL)
    {}

    llvm::Value* putRet(Type* dty, DValue* dval);
    llvm::Value* getRet(Type* dty, DValue* dval);

    llvm::Value* putParam(Type* dty, size_t idx, DValue* dval);
    llvm::Value* getParam(Type* dty, size_t idx, DValue* dval);
    void getParam(Type* dty, size_t idx, DValue* dval, llvm::Value* lval);
};

#endif
