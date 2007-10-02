#ifndef LLVMDC_GEN_ELEM_H
#define LLVMDC_GEN_ELEM_H

#include "llvm/Value.h"

#include "root.h"
#include "declaration.h"
#include "aggregate.h"

// represents a value. be it a constant literal, a variable etc.
// maintains all the information for doing load/store appropriately
struct elem : Object
{
    enum {
        NONE,
        VAR,
        VAL,
        FUNC,
        CONST,
        NUL,
        REF,
        SLICE
    };

public:
    elem();

    llvm::Value* mem;
    llvm::Value* val;
    llvm::Value* arg;
    int type;
    bool inplace;
    bool field;
    unsigned callconv;

    VarDeclaration* vardecl;
    FuncDeclaration* funcdecl;

    llvm::Value* getValue();
    //llvm::Value* getMemory();

    bool isNull()   {return !(mem || val);}
};

#endif // LLVMDC_GEN_ELEM_H
