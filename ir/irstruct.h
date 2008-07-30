#ifndef LLVMDC_IR_IRSTRUCT_H
#define LLVMDC_IR_IRSTRUCT_H

#include "ir/ir.h"

#include <vector>
#include <map>

struct IrInterface : IrBase
{
    BaseClass* base;
    ClassDeclaration* decl;

    llvm::PATypeHolder* vtblTy;
    LLConstant* vtblInit;
    LLGlobalVariable* vtbl;

    const LLStructType* infoTy;
    LLConstantStruct* infoInit;
    LLConstant* info;

    int index;

    IrInterface(BaseClass* b);
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
        const LLType* type;
        LLConstant* init;

        Offset(VarDeclaration* v, const LLType* ty)
        : var(v), type(ty), init(NULL) {}
    };

    typedef std::multimap<unsigned, Offset> OffsetMap;
    typedef std::vector<VarDeclaration*> VarDeclVector;
    typedef std::map<ClassDeclaration*, IrInterface*> InterfaceMap;
    typedef InterfaceMap::iterator InterfaceMapIter;
    typedef std::vector<IrInterface*> InterfaceVector;
    typedef InterfaceVector::iterator InterfaceVectorIter;

public:
    IrStruct(Type*);
    virtual ~IrStruct();

    Type* type;
    llvm::PATypeHolder recty;
    OffsetMap offsets;
    VarDeclVector defaultFields;

    InterfaceMap interfaceMap;
    InterfaceVector interfaceVec;
    const llvm::ArrayType* interfaceInfosTy;
    LLGlobalVariable* interfaceInfos;

    bool defined;
    bool constinited;

    LLGlobalVariable* vtbl;
#if OPAQUE_VTBLS
    LLConstant* constVtbl;
#else
    LLConstantStruct* constVtbl;
#endif
    LLGlobalVariable* init;
    LLConstant* constInit;
    LLGlobalVariable* classInfo;
    LLConstant* constClassInfo;
    bool hasUnions;
    DUnion* dunion;
    bool classDeclared;
    bool classDefined;

    bool packed; // true for: align(1) struct S { ... }

    LLGlobalVariable* dwarfComposite;
};

#endif
