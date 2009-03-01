#ifndef __LDC_GEN_ABI_H__
#define __LDC_GEN_ABI_H__

#include <vector>

struct Type;
namespace llvm
{
    class Type;
    class Value;
}

// return rewrite rule
struct ABIRetRewrite
{
    // get original value from rewritten one
    virtual LLValue* get(LLValue* v) = 0;

    // rewrite original value
    virtual LLValue* put(LLValue* v) = 0;

    // returns target type of this rewrite
    virtual const LLType* type(const LLType* t) = 0;

    // test if rewrite applies
    virtual bool test(TypeFunction* tf) = 0;
};


// interface called by codegen
struct TargetABI
{
    static TargetABI* getTarget();

    TargetABI();

    const llvm::Type* getRetType(TypeFunction* tf, const llvm::Type* t);
    llvm::Value* getRet(TypeFunction* tf, llvm::Value* v);
    llvm::Value* putRet(TypeFunction* tf, llvm::Value* v);

    virtual bool returnInArg(TypeFunction* t) = 0;

protected:
    std::vector<ABIRetRewrite*> retOps;
    ABIRetRewrite* findRetRewrite(TypeFunction* tf);
};

#endif
