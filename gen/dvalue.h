#ifndef LLVMDC_GEN_DVALUE_H
#define LLVMDC_GEN_DVALUE_H

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
struct DThisValue;
struct DFuncValue;
struct DSliceValue;
struct DArrayLenValue;
struct DLRValue;
struct DComplexValue;

// base class for d-values
struct DValue : Object
{
    virtual Type* getType() = 0;

    virtual llvm::Value* getLVal() { assert(0); return 0; }
    virtual llvm::Value* getRVal() { assert(0); return 0; }

    virtual DImValue* isIm() { return NULL; }
    virtual DConstValue* isConst() { return NULL; }
    virtual DNullValue* isNull() { return NULL; }
    virtual DVarValue* isVar() { return NULL; }
    virtual DFieldValue* isField() { return NULL; }
    virtual DThisValue* isThis() { return NULL; }
    virtual DSliceValue* isSlice() { return NULL; }
    virtual DFuncValue* isFunc() { return NULL; }
    virtual DArrayLenValue* isArrayLen() { return NULL; }
    virtual DComplexValue* isComplex() { return NULL; }
    virtual DLRValue* isLRValue() { return NULL; }

    virtual bool inPlace() { return false; }

protected:
    DValue() {}
    DValue(const DValue&) { }
    DValue& operator=(const DValue&) { return *this; }
};

// immediate d-value
struct DImValue : DValue
{
    Type* type;
    llvm::Value* val;
    bool inplace;

    DImValue(Type* t, llvm::Value* v, bool in_place = false) { type = t; val = v; inplace = in_place; }

    virtual llvm::Value* getRVal() { assert(val); return val; }

    virtual Type* getType() { assert(type); return type; }
    virtual DImValue* isIm() { return this; }

    virtual bool inPlace() { return inplace; }
};

// constant d-value
struct DConstValue : DValue
{
    Type* type;
    llvm::Constant* c;

    DConstValue(Type* t, llvm::Constant* con) { type = t; c = con; }

    virtual llvm::Value* getRVal();

    virtual Type* getType() { assert(type); return type; }
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
    Type* type;
    VarDeclaration* var;
    llvm::Value* val;
    llvm::Value* rval;
    bool lval;

    DVarValue(VarDeclaration* vd, llvm::Value* llvmValue, bool lvalue);
    DVarValue(Type* vd, llvm::Value* lv, llvm::Value* rv);
    DVarValue(Type* t, llvm::Value* llvmValue, bool lvalue);

    virtual llvm::Value* getLVal();
    virtual llvm::Value* getRVal();

    virtual Type* getType() { assert(type); return type; }
    virtual DVarValue* isVar() { return this; }
};

// field d-value
struct DFieldValue : DVarValue
{
    DFieldValue(Type* t, llvm::Value* llvmValue, bool l) : DVarValue(t, llvmValue, l) {}
    virtual DFieldValue* isField() { return this; }
};

// this d-value
struct DThisValue : DVarValue
{
    DThisValue(VarDeclaration* vd, llvm::Value* llvmValue) : DVarValue(vd, llvmValue, true) {}
    virtual DThisValue* isThis() { return this; }
};

// array length d-value
struct DArrayLenValue : DVarValue
{
    DArrayLenValue(Type* t, llvm::Value* llvmValue) : DVarValue(t, llvmValue, true) {}
    virtual DArrayLenValue* isArrayLen() { return this; }
};

// slice d-value
struct DSliceValue : DValue
{
    Type* type;
    llvm::Value* len;
    llvm::Value* ptr;

    DSliceValue(Type* t, llvm::Value* l, llvm::Value* p) { type=t; ptr=p; len=l; }

    virtual Type* getType() { assert(type); return type; }
    virtual DSliceValue* isSlice() { return this; }
};

// function d-value
struct DFuncValue : DValue
{
    Type* type;
    FuncDeclaration* func;
    llvm::Value* val;
    llvm::Value* vthis;
    unsigned cc;

    DFuncValue(FuncDeclaration* fd, llvm::Value* v, llvm::Value* vt = 0);

    virtual llvm::Value* getLVal();
    virtual llvm::Value* getRVal();

    virtual Type* getType() { assert(type); return type; }
    virtual DFuncValue* isFunc() { return this; }
};

// l-value and r-value pair d-value
struct DLRValue : DValue
{
    Type* ltype;
    llvm::Value* lval;
    Type* rtype;
    llvm::Value* rval;

    DLRValue(Type* lt, llvm::Value* l, Type* rt, llvm::Value* r) {
        ltype = lt;
        lval = l;
        rtype = rt;
        rval = r;
    }

    virtual llvm::Value* getLVal() { assert(lval); return lval; }
    virtual llvm::Value* getRVal() { assert(rval); return rval; }

    Type* getLType() { return ltype; }
    Type* getRType() { return rtype; }
    virtual Type* getType() { return getRType(); }
    virtual DLRValue* isLRValue() { return this; }
};

// complex number immediate d-value (much like slice)
struct DComplexValue : DValue
{
    Type* type;
    llvm::Value* re;
    llvm::Value* im;

    DComplexValue(Type* t, llvm::Value* r, llvm::Value* i) {
        type = t;
        re = r;
        im = i;
    }

    virtual Type* getType() { assert(type); return type; }
    virtual DComplexValue* isComplex() { return this; }
};

#endif // LLVMDC_GEN_DVALUE_H
