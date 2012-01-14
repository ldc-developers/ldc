#ifndef __LDC_IR_IRTYPE_H__
#define __LDC_IR_IRTYPE_H__

#include "llvm/Type.h"

//////////////////////////////////////////////////////////////////////////////

// forward declarations

struct Type;

class IrTypeAggr;
class IrTypeArray;
class IrTypeBasic;
class IrTypeClass;
class IrTypeDelegate;
class IrTypeFunction;
class IrTypePointer;
class IrTypeSArray;
class IrTypeStruct;

//////////////////////////////////////////////////////////////////////////////

/// Base class for IrTypeS.
class IrType
{
public:
    ///
    IrType(Type* dt, llvm::Type* lt);

    ///
    virtual IrTypeAggr* isAggr()        { return NULL; }
    ///
    virtual IrTypeArray* isArray()      { return NULL; }
    ///
    virtual IrTypeBasic* isBasic()      { return NULL; }
    ///
    virtual IrTypeClass* isClass()      { return NULL; }
    ///
    virtual IrTypeDelegate* isDelegate(){ return NULL; }
    ///
    virtual IrTypeFunction* isFunction(){ return NULL; }
    ///
    virtual IrTypePointer* isPointer()  { return NULL; }
    ///
    virtual IrTypeSArray* isSArray()    { return NULL; }
    ///
    virtual IrTypeStruct* isStruct()    { return NULL; }

    ///
    Type* getD()                        { return dtype; }
    ///
    virtual llvm::Type* get()           { return type; }
    ///
    llvm::Type* getType()               { return type; }

    ///
    virtual llvm::Type* buildType() = 0;

protected:
    ///
    Type* dtype;

    /// LLVM type.
    llvm::Type* type;
};

//////////////////////////////////////////////////////////////////////////////

/// IrType for basic D types.
class IrTypeBasic : public IrType
{
public:
    ///
    IrTypeBasic(Type* dt);

    ///
    IrTypeBasic* isBasic()          { return this; }

    ///
    llvm::Type* buildType();

protected:
    ///
    LLType* getComplexType(llvm::LLVMContext& ctx, LLType* type);
    ///
    llvm::Type* basic2llvm(Type* t);
};

//////////////////////////////////////////////////////////////////////////////

/// IrType from pointers.
class IrTypePointer : public IrType
{
public:
    ///
    IrTypePointer(Type* dt);

    ///
    IrTypePointer* isPointer()      { return this; }

    ///
    llvm::Type* buildType();

protected:
    ///
    llvm::Type* pointer2llvm(Type* t);
    ///
    llvm::Type* null2llvm(Type* t);
};

//////////////////////////////////////////////////////////////////////////////

/// IrType for static arrays
class IrTypeSArray : public IrType
{
public:
    ///
    IrTypeSArray(Type* dt);

    ///
    IrTypeSArray* isSArray()  { return this; }

    ///
    llvm::Type* buildType();

protected:
    ///
    llvm::Type* sarray2llvm(Type* t);

    /// Dimension.
    uint64_t dim;
};

//////////////////////////////////////////////////////////////////////////////

/// IrType for dynamic arrays
class IrTypeArray : public IrType
{
public:
    ///
    IrTypeArray(Type* dt);

    ///
    IrTypeArray* isArray()  { return this; }

    ///
    llvm::Type* buildType();

protected:
    ///
    llvm::Type* array2llvm(Type* t);
};

//////////////////////////////////////////////////////////////////////////////

#endif
