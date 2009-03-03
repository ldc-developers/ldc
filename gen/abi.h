#ifndef __LDC_GEN_ABI_H__
#define __LDC_GEN_ABI_H__

#include <vector>

struct Type;
struct IrFuncTyArg;
namespace llvm
{
    class Type;
    class Value;
}

// return rewrite rule
struct ABIRewrite
{
    // get original value from rewritten one
    virtual LLValue* get(Type* dty, LLValue* v) = 0;

    // rewrite original value
    virtual LLValue* put(Type* dty, LLValue* v) = 0;

    // returns target type of this rewrite
    virtual const LLType* type(Type* dty, const LLType* t) = 0;
};

// interface called by codegen
struct TargetABI
{
    static TargetABI* getTarget();

    virtual bool returnInArg(TypeFunction* tf) = 0;
    virtual bool passByVal(Type* t) = 0;

    virtual void rewriteFunctionType(TypeFunction* t) = 0;
};

#endif
