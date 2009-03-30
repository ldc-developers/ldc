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

// base class for d-values
struct DValue : Object
{
    Type* type;
    DValue(Type* ty) : type(ty) {}

    Type*& getType() { assert(type); return type; }

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

protected:
    DValue() {}
    DValue(const DValue&) { }
    DValue& operator=(const DValue& other) { type = other.type; return *this; }
};

// immediate d-value
struct DImValue : DValue
{
    LLValue* val;

    DImValue(Type* t, LLValue* v) : DValue(t), val(v) { }

    virtual LLValue* getRVal() { assert(val); return val; }

    virtual DImValue* isIm() { return this; }
};

// constant d-value
struct DConstValue : DValue
{
    LLConstant* c;

    DConstValue(Type* t, LLConstant* con) : DValue(t), c(con) {}

    virtual LLValue* getRVal();

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
    VarDeclaration* var;
    LLValue* val;

    DVarValue(Type* t, VarDeclaration* vd, LLValue* llvmValue);
    DVarValue(Type* t, LLValue* llvmValue);

    virtual bool isLVal() { return true; }
    virtual LLValue* getLVal();
    virtual LLValue* getRVal();

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
    LLValue* len;
    LLValue* ptr;

    DSliceValue(Type* t, LLValue* l, LLValue* p) : DValue(t), len(l), ptr(p) {}

    virtual LLValue* getRVal();

    virtual DSliceValue* isSlice() { return this; }
};

// function d-value
struct DFuncValue : DValue
{
    FuncDeclaration* func;
    LLValue* val;
    LLValue* vthis;

    DFuncValue(FuncDeclaration* fd, LLValue* v, LLValue* vt = 0);

    virtual LLValue* getRVal();

    virtual DFuncValue* isFunc() { return this; }
};

#endif // LDC_GEN_DVALUE_H
