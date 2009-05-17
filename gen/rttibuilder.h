#ifndef __LDC_GEN_RTTIBUILDER_H__
#define __LDC_GEN_RTTIBUILDER_H__

#include "llvm/Constant.h"
#include "llvm/ADT/SmallVector.h"

struct ClassDeclaration;
struct TypeClass;

struct IrStruct;

struct TypeInfoBuilder
{
    ClassDeclaration* base;
    TypeClass* basetype;
    IrStruct* baseir;

    // 10 is enough for any D1 typeinfo
    llvm::SmallVector<llvm::Constant*, 10> inits;

    TypeInfoBuilder(ClassDeclaration* base_class);

    void push(llvm::Constant* C);
    void push_null_vp();
    void push_typeinfo(Type* t);
    void push_classinfo(ClassDeclaration* cd);
    void push_string(const char* str);
    void push_null_void_array();
    void push_void_array(size_t dim, llvm::Constant* ptr);
    void push_void_array(llvm::Constant* CI, Type* valtype, Dsymbol* sym);
    void finalize(IrGlobal* tid);
};

#endif
