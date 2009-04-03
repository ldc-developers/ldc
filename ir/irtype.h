#ifndef __LDC_IR_IRTYPE_H__
#define __LDC_IR_IRTYPE_H__

#include "llvm/Type.h"

//////////////////////////////////////////////////////////////////////////////

// forward declarations

struct Type;

class IrTypeArray;
class IrTypeBasic;
class IrTypePointer;
class IrTypeSArray;

//////////////////////////////////////////////////////////////////////////////

/// Base class for IrTypeS.
class IrType
{
public:
    ///
    IrType(Type* dt, const llvm::Type* lt);

    ///
    Type* getD()                    { return dtype; }

    ///
    const llvm::Type* get()         { return pa.get(); }

    ///
    virtual IrTypeArray* isArray()      { return NULL; }
    ///
    virtual IrTypeBasic* isBasic()      { return NULL; }
    ///
    virtual IrTypePointer* isPointer()  { return NULL; }
    ///
    virtual IrTypeSArray* isSArray()    { return NULL; }

protected:
    ///
    Type* dtype;

    /// LLVM type holder.
    llvm::PATypeHolder pa;
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

protected:
    ///
    const llvm::Type* basic2llvm(Type* t);
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

protected:
    ///
    const llvm::Type* pointer2llvm(Type* t);
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

protected:
    ///
    const llvm::Type* sarray2llvm(Type* t);

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

protected:
    ///
    const llvm::Type* array2llvm(Type* t);
};

//////////////////////////////////////////////////////////////////////////////

#endif
