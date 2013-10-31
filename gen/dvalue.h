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

#include "root.h"
#include <cassert>

class Type;
class Dsymbol;
class VarDeclaration;
class FuncDeclaration;

namespace llvm
{
    class Value;
    class Type;
    class Constant;
}

class DImValue;
class DConstValue;
class DNullValue;
class DVarValue;
class DFieldValue;
class DFuncValue;
class DSliceValue;

// base class for d-values
class DValue
{
public:
    Type* type;
    DValue(Type* ty) : type(ty) {}
    virtual ~DValue() {}

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
class DImValue : public DValue
{
public:
    DImValue(Type* t, llvm::Value* v) : DValue(t), val(v) { }

    virtual llvm::Value* getRVal() { assert(val); return val; }

    virtual DImValue* isIm() { return this; }

protected:
    llvm::Value* val;
};

// constant d-value
class DConstValue : public DValue
{
public:
    DConstValue(Type* t, llvm::Constant* con) : DValue(t), c(con) {}

    virtual llvm::Value* getRVal();

    virtual DConstValue* isConst() { return this; }

    llvm::Constant* c;
};

// null d-value
class DNullValue : public DConstValue
{
public:
    DNullValue(Type* t, llvm::Constant* con) : DConstValue(t,con) {}
    virtual DNullValue* isNull() { return this; }
};

// variable d-value
class DVarValue : public DValue
{
public:
    DVarValue(Type* t, VarDeclaration* vd, llvm::Value* llvmValue);
    DVarValue(Type* t, llvm::Value* llvmValue);

    virtual bool isLVal() { return true; }
    virtual llvm::Value* getLVal();
    virtual llvm::Value* getRVal();

    /// Returns the underlying storage for special internal ref variables.
    /// Illegal to call on any other value.
    virtual llvm::Value* getRefStorage();

    virtual DVarValue* isVar() { return this; }

    VarDeclaration* var;
protected:
    llvm::Value* val;
};

// field d-value
class DFieldValue : public DVarValue
{
public:
    DFieldValue(Type* t, llvm::Value* llvmValue) : DVarValue(t, llvmValue) {}
    virtual DFieldValue* isField() { return this; }
};

// slice d-value
class DSliceValue : public DValue
{
public:
    DSliceValue(Type* t, llvm::Value* l, llvm::Value* p) : DValue(t), len(l), ptr(p) {}

    virtual llvm::Value* getRVal();

    virtual DSliceValue* isSlice() { return this; }

    llvm::Value* len;
    llvm::Value* ptr;
};

// function d-value
class DFuncValue : public DValue
{
public:
    DFuncValue(Type *t, FuncDeclaration* fd, llvm::Value* v, llvm::Value* vt = 0);
    DFuncValue(FuncDeclaration* fd, llvm::Value* v, llvm::Value* vt = 0);

    virtual llvm::Value* getRVal();

    virtual DFuncValue* isFunc() { return this; }

    FuncDeclaration* func;
    llvm::Value* val;
    llvm::Value* vthis;
};

#endif // LDC_GEN_DVALUE_H
