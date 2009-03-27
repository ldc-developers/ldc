#ifndef LDC_IR_IRSTRUCT_H
#define LDC_IR_IRSTRUCT_H

#include "ir/ir.h"

#include <vector>
#include <map>

struct IrInterface;

void addZeros(std::vector<const llvm::Type*>& inits, size_t pos, size_t offset);
void addZeros(std::vector<llvm::Constant*>& inits, size_t pos, size_t offset);
void addZeros(std::vector<llvm::Value*>& inits, size_t pos, size_t offset);

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// represents a struct or class
// it is used during codegen to hold all the vital info we need
struct IrStruct : IrBase
{
    /////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////

    typedef std::vector<VarDeclaration*> VarDeclVector;

    typedef std::map<ClassDeclaration*, IrInterface*>   InterfaceMap;
    typedef InterfaceMap::iterator                      InterfaceMapIter;

    typedef std::vector<IrInterface*> InterfaceVector;
    typedef InterfaceVector::iterator InterfaceVectorIter;

    // vector of LLVM types
    typedef std::vector<const llvm::Type*> TypeVector;

    /////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////

    // Anon represents an anonymous struct union block inside an aggregate
    // during LLVM type construction.
    struct Anon
    {
        bool isunion;
        Anon* parent;

        TypeVector types;

        Anon(bool IsUnion, Anon* par) : isunion(IsUnion), parent(par) {}
    };

    /////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////

    /// ctor
    IrStruct(AggregateDeclaration* agg);

    /// dtor
    virtual ~IrStruct();

    /////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////

    /// push an anonymous struct/union
    void pushAnon(bool isunion);

    /// pops an anonymous struct/union
    void popAnon();

    /// adds field
    void addVar(VarDeclaration* var);

    /////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////

    /// build the aggr type
    const LLType* build();

    /// put the aggr initializers in a vector
    void buildDefaultConstInit(std::vector<llvm::Constant*>& inits);

    /// ditto - but also builds the constant struct, for convenience
    LLConstant* buildDefaultConstInit();

    /////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////

    // the D aggregate
    AggregateDeclaration* aggrdecl;

    // vector of VarDeclarations in this aggregate
    VarDeclVector varDecls;

    // vector of VarDeclarations that contribute to the default initializer
    VarDeclVector defVars;

    // true if the default initializer has been built
    bool defaultFound;

    // top element
    Anon* anon;

    // toplevel types in this aggr
    TypeVector types;

    // current index
    // always the same as types.size()
    size_t index; 

    // aggregate D type
    Type* type;

    // class vtable type
    llvm::PATypeHolder vtblTy;
    llvm::PATypeHolder vtblInitTy;

    // initializer type opaque (type of global matches initializer, not formal type)
    llvm::PATypeHolder initOpaque;
    llvm::PATypeHolder classInfoOpaque;

    // map/vector of interfaces implemented
    InterfaceMap interfaceMap;
    InterfaceVector interfaceVec;

    // interface info array global
    LLGlobalVariable* interfaceInfos;

    // ...
    bool defined;
    bool constinited;

    // vtbl global and initializer
    LLGlobalVariable* vtbl;
    LLConstant* constVtbl;

    // static initializers global and constant
    LLGlobalVariable* init;
    LLConstant* constInit;

    // classinfo global and initializer constant
    LLGlobalVariable* classInfo;
    LLConstant* constClassInfo;
    bool classInfoDeclared;
    bool classInfoDefined;
    // vector of interfaces that should be put in ClassInfo.interfaces
    InterfaceVector classInfoInterfaces;

    // align(1) struct S { ... }
    bool packed;

    // composite type debug description
    llvm::DICompositeType diCompositeType;

    std::vector<VarDeclaration*> staticVars;
};

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// represents interface implemented by a class
struct IrInterface : IrBase
{
    BaseClass* base;
    ClassDeclaration* decl;

    llvm::PATypeHolder vtblInitTy;

    LLConstant* vtblInit;
    LLGlobalVariable* vtbl;
    Array vtblDecls; // array of FuncDecls that make up the vtbl

    const LLStructType* infoTy;
    LLConstant* infoInit;
    LLConstant* info;

    size_t index;

    IrInterface(BaseClass* b);
};

#endif
