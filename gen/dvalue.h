//===-- gen/dvalue.h - D value abstractions ---------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// These classes are used for generating the IR. They encapsulate an LLVM value
// together with a D type and provide an uniform interface for the most common
// operations. When more specialize handling is necessary, they hold enough
// information to do-the-right-thing (TM).
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_DVALUE_H
#define LDC_GEN_DVALUE_H

#include <cassert>
#include "root.h"

struct Type;
struct Dsymbol;
struct VarDeclaration;
struct FuncDeclaration;

namespace llvm
{
    class Value;
    class Type;
    class Constant;
}

struct DImValue;
struct DConstValue;
struct DNullValue;
struct DVarValue;
struct DFieldValue;
struct DFuncValue;
struct DSliceValue;

// base class for d-values
struct DValue : Object
{
    Type* type;
    DValue(Type* ty) : type(ty) {}

    Type*& getType() { assert(type); return type; }

    virtual llvm::Value* getLVal() { assert(0); return 0; }
    virtual llvm::Value* getRVal() { assert(0); return 0; }

    virtual bool isLVal() { return false; }

    virtual DImValue* isIm() { return NULL; }
    virtual DConstValue* isConst() { return NULL; }
    virtual DNullValue* isNull() { return NULL; }
    virtual DVarValue* isVar() { return NULL; }
    virtual DFieldValue* isField() { return NULL; }
    virtual DSliceValue* isSlice() { return NULL; }
    virtual DFuncValue* isFunc() { return NULL; }
protected:
    DValue() {}
    DValue(const DValue&) { }
    DValue& operator=(const DValue& other) { type = other.type; return *this; }
};

// immediate d-value
struct DImValue : DValue
{
    llvm::Value* val;

    DImValue(Type* t, llvm::Value* v) : DValue(t), val(v) { }

    virtual llvm::Value* getRVal() { assert(val); return val; }

    virtual DImValue* isIm() { return this; }
};

// constant d-value
struct DConstValue : DValue
{
    llvm::Constant* c;

    DConstValue(Type* t, llvm::Constant* con) : DValue(t), c(con) {}

    virtual llvm::Value* getRVal();

    virtual DConstValue* isConst() { return this; }
};

// null d-value
struct DNullValue : DConstValue
{
    DNullValue(Type* t, llvm::Constant* con) : DConstValue(t,con) {}
    virtual DNullValue* isNull() { return this; }
};

// variable d-value
struct DVarValue : DValue
{
    VarDeclaration* var;
    llvm::Value* val;

    DVarValue(Type* t, VarDeclaration* vd, llvm::Value* llvmValue);
    DVarValue(Type* t, llvm::Value* llvmValue);

    virtual bool isLVal() { return true; }
    virtual llvm::Value* getLVal();
    virtual llvm::Value* getRVal();

    /// Returns the underlying storage for special internal ref variables.
    /// Illegal to call on any other value.
    virtual llvm::Value* getRefStorage();

    virtual DVarValue* isVar() { return this; }
};

// field d-value
struct DFieldValue : DVarValue
{
    DFieldValue(Type* t, llvm::Value* llvmValue) : DVarValue(t, llvmValue) {}
    virtual DFieldValue* isField() { return this; }
};

// slice d-value
struct DSliceValue : DValue
{
    llvm::Value* len;
    llvm::Value* ptr;

    DSliceValue(Type* t, llvm::Value* l, llvm::Value* p) : DValue(t), len(l), ptr(p) {}

    virtual llvm::Value* getRVal();

    virtual DSliceValue* isSlice() { return this; }
};

// function d-value
struct DFuncValue : DValue
{
    FuncDeclaration* func;
    llvm::Value* val;
    llvm::Value* vthis;

    DFuncValue(FuncDeclaration* fd, llvm::Value* v, llvm::Value* vt = 0);

    virtual llvm::Value* getRVal();

    virtual DFuncValue* isFunc() { return this; }
};

#endif // LDC_GEN_DVALUE_H
