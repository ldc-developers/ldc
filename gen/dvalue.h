#ifndef LDC_GEN_DVALUE_H
#define LDC_GEN_DVALUE_H

/*
These classes are used for generating the IR. They encapsulate D values and
provide a common interface to the most common operations. When more specialized
handling is necessary, they hold enough information to do-the-right-thing (TM)
*/

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
struct DArrayLenValue;
struct DLRValue;

// base class for d-values
struct DValue : Object
{
    virtual Type*& getType() = 0;

    virtual LLValue* getLVal() { assert(0); return 0; }
    virtual LLValue* getRVal() { assert(0); return 0; }

    virtual bool isLVal() { return false; }

    virtual DImValue* isIm() { return NULL; }
    virtual DConstValue* isConst() { return NULL; }
    virtual DNullValue* isNull() { return NULL; }
    virtual DVarValue* isVar() { return NULL; }
    virtual DFieldValue* isField() { return NULL; }
    virtual DSliceValue* isSlice() { return NULL; }
    virtual DFuncValue* isFunc() { return NULL; }
    virtual DArrayLenValue* isArrayLen() { return NULL; }
    virtual DLRValue* isLRValue() { return NULL; }

protected:
    DValue() {}
    DValue(const DValue&) { }
    DValue& operator=(const DValue&) { return *this; }
};

// immediate d-value
struct DImValue : DValue
{
    Type* type;
    LLValue* val;

    DImValue(Type* t, LLValue* v) : type(t), val(v) { }

    virtual LLValue* getRVal() { assert(val); return val; }

    virtual Type*& getType() { assert(type); return type; }
    virtual DImValue* isIm() { return this; }
};

// constant d-value
struct DConstValue : DValue
{
    Type* type;
    LLConstant* c;

    DConstValue(Type* t, LLConstant* con) { type = t; c = con; }

    virtual LLValue* getRVal();

    virtual Type*& getType() { assert(type); return type; }
    virtual DConstValue* isConst() { return this; }
};

// null d-value
struct DNullValue : DConstValue
{
    DNullValue(Type* t, LLConstant* con) : DConstValue(t,con) {}
    virtual DNullValue* isNull() { return this; }
};

// variable d-value
struct DVarValue : DValue
{
    Type* type;
    VarDeclaration* var;
    LLValue* val;

    DVarValue(Type* t, VarDeclaration* vd, LLValue* llvmValue);
    DVarValue(Type* t, LLValue* llvmValue);

    virtual bool isLVal() { return true; }
    virtual LLValue* getLVal();
    virtual LLValue* getRVal();

    virtual Type*& getType() { assert(type); return type; }
    virtual DVarValue* isVar() { return this; }
};

// field d-value
struct DFieldValue : DVarValue
{
    DFieldValue(Type* t, LLValue* llvmValue) : DVarValue(t, llvmValue) {}
    virtual DFieldValue* isField() { return this; }
};

// slice d-value
struct DSliceValue : DValue
{
    Type* type;
    LLValue* len;
    LLValue* ptr;

    DSliceValue(Type* t, LLValue* l, LLValue* p) { type=t; ptr=p; len=l; }

    virtual Type*& getType() { assert(type); return type; }
    virtual DSliceValue* isSlice() { return this; }
};

// function d-value
struct DFuncValue : DValue
{
    Type* type;
    FuncDeclaration* func;
    LLValue* val;
    LLValue* vthis;

    DFuncValue(FuncDeclaration* fd, LLValue* v, LLValue* vt = 0);

    virtual LLValue* getRVal();

    virtual Type*& getType() { assert(type); return type; }
    virtual DFuncValue* isFunc() { return this; }
};

// l-value and r-value pair d-value
struct DLRValue : DValue
{
    DValue* lvalue;
    DValue* rvalue;

    DLRValue(DValue* lval, DValue* rval) {
        lvalue = lval;
        rvalue = rval;
    }

    virtual bool isLVal() { return true; }
    virtual LLValue* getLVal() { return lvalue->isLVal() ? lvalue->getLVal() : lvalue->getRVal(); }
    virtual LLValue* getRVal() { return rvalue->getRVal(); }

    Type*& getLType();
    Type*& getRType() { return rvalue->getType(); }
    virtual Type*& getType() { return getRType(); }
    virtual DLRValue* isLRValue() { return this; }
};

#endif // LDC_GEN_DVALUE_H
