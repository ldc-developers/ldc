#ifndef __LDC_GEN_RTTIBUILDER_H__
#define __LDC_GEN_RTTIBUILDER_H__

#include "llvm/Constant.h"
#include "llvm/ADT/SmallVector.h"

struct ClassDeclaration;
struct TypeClass;
struct Type;

struct IrStruct;

struct RTTIBuilder
{
    ClassDeclaration* base;
    TypeClass* basetype;
    IrStruct* baseir;

    // 10 is enough for any D1 typeinfo
    llvm::SmallVector<llvm::Constant*, 10> inits;

    RTTIBuilder(ClassDeclaration* base_class);

    void push(llvm::Constant* C);
    void push_null_vp();
    void push_null_void_array();
    void push_uint(unsigned u);
    void push_size(uint64_t s);
    void push_string(const char* str);
    void push_typeinfo(Type* t);
    void push_classinfo(ClassDeclaration* cd);
    void push_funcptr(FuncDeclaration* fd);
    void push_void_array(uint64_t dim, llvm::Constant* ptr);
    void push_void_array(llvm::Constant* CI, Type* valtype, Dsymbol* mangle_sym);
    void push_array(llvm::Constant* CI, uint64_t dim, Type* valtype, Dsymbol* mangle_sym);

    /// Creates the initializer constant and assigns it to the global.
    void finalize(IrGlobal* tid);
};

#endif
