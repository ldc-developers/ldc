#ifndef LLVMDC_IR_IRSTRUCT_H
#define LLVMDC_IR_IRSTRUCT_H

#include "ir/ir.h"

#include <vector>
#include <map>

struct IrInterface : IrBase
{
    BaseClass* base;
    ClassDeclaration* decl;

    const llvm::StructType* vtblTy;
    llvm::ConstantStruct* vtblInit;
    llvm::GlobalVariable* vtbl;

    const llvm::StructType* infoTy;
    llvm::ConstantStruct* infoInit;
    llvm::Constant* info;

    int index;

    IrInterface(BaseClass* b, const llvm::StructType* vt);
    ~IrInterface();
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// represents a struct or class
struct IrStruct : IrBase
{
    struct Offset
    {
        VarDeclaration* var;
        const llvm::Type* type;
        llvm::Constant* init;

        Offset(VarDeclaration* v, const llvm::Type* ty)
        : var(v), type(ty), init(NULL) {}
    };

    struct InterCmp
    {
        bool operator()(ClassDeclaration* lhs, ClassDeclaration* rhs) const
        {
            return strcmp(lhs->toPrettyChars(), rhs->toPrettyChars()) < 0;
        }
    };

    typedef std::multimap<unsigned, Offset> OffsetMap;
    typedef std::vector<VarDeclaration*> VarDeclVector;
    typedef std::map<ClassDeclaration*, IrInterface*, InterCmp> InterfaceMap;
    typedef InterfaceMap::iterator InterfaceIter;

public:
    IrStruct(Type*);
    virtual ~IrStruct();

    Type* type;
    llvm::PATypeHolder recty;
    OffsetMap offsets;
    VarDeclVector defaultFields;

    InterfaceMap interfaces;
    const llvm::ArrayType* interfaceInfosTy;
    llvm::GlobalVariable* interfaceInfos;

    bool defined;
    bool constinited;

    llvm::GlobalVariable* vtbl;
    llvm::ConstantStruct* constVtbl;
    llvm::GlobalVariable* init;
    llvm::Constant* constInit;
    llvm::GlobalVariable* classInfo;
    llvm::Constant* constClassInfo;
    bool hasUnions;
    DUnion* dunion;
    bool classDeclared;
    bool classDefined;
};

#endif
