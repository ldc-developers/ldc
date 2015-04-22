//===-- ir/irtype.h - IrType base and primitive types -----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The types derived from IrType are used to attach LLVM type information and
// other codegen metadata (e.g. for vtbl resolution) to frontend Types.
//
//===----------------------------------------------------------------------===//


#ifndef __LDC_IR_IRTYPE_H__
#define __LDC_IR_IRTYPE_H__

#include "ir/irfuncty.h"

//////////////////////////////////////////////////////////////////////////////

// forward declarations

namespace llvm
{
    class LLVMContext;
    class Type;
}

class Type;

class IrTypeAggr;
class IrTypeArray;
class IrTypeBasic;
class IrTypeClass;
class IrTypeDelegate;
class IrTypeFunction;
class IrTypePointer;
class IrTypeSArray;
class IrTypeStruct;
class IrTypeVector;

//////////////////////////////////////////////////////////////////////////////

/// Code generation state/metadata for D types. The mapping from IrType to
/// Type is injective but not surjective.
///
/// Derived classes should be created using their static get() methods, which
/// makes sure that uniqueness is preserved in the face of forward references.
/// Note that the get() methods expect the IrType of the passed type/symbol to
/// be not yet set.
///
/// This could be altered to just return the existing IrType in order to bring
/// the API entirely in line with the LLVM type get() methods. It has not been
/// changed so far since currently all clients use the DtoType wrapper rather
/// than handling IrType instances directly, and keeping it this way allows to
/// easily check for uniqueness violations in the face of forward references.
/// TODO: Implement the described changes (now that the forward reference
/// handling logic seems to work correctly) and get rid of the "no-op" DtoType
/// calls in IrAggr, ... that only exist for their side effect.
class IrType
{
public:
    virtual ~IrType() {}

    ///
    virtual IrTypeAggr* isAggr()        { return 0; }
    ///
    virtual IrTypeArray* isArray()      { return 0; }
    ///
    virtual IrTypeBasic* isBasic()      { return 0; }
    ///
    virtual IrTypeClass* isClass()      { return 0; }
    ///
    virtual IrTypeDelegate* isDelegate(){ return 0; }
    ///
    virtual IrTypeFunction* isFunction(){ return 0; }
    ///
    virtual IrTypePointer* isPointer()  { return 0; }
    ///
    virtual IrTypeSArray* isSArray()    { return 0; }
    ///
    virtual IrTypeStruct* isStruct()    { return 0; }
    ///
    virtual IrTypeVector* isVector()    { return 0; }

    ///
    Type* getDType()                    { return dtype; }
    ///
    virtual llvm::Type* getLLType()     { return type; }

    ///
    virtual IrFuncTy &getIrFuncTy();

protected:
    ///
    IrType(Type* dt, llvm::Type* lt);

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
    static IrTypeBasic* get(Type* dt);

    ///
    IrTypeBasic* isBasic()          { return this; }

protected:
    ///
    IrTypeBasic(Type* dt);
    ///
    static llvm::Type* getComplexType(llvm::LLVMContext& ctx, llvm::Type* type);
    ///
    static llvm::Type* basic2llvm(Type* t);
};

//////////////////////////////////////////////////////////////////////////////

/// IrType from pointers.
class IrTypePointer : public IrType
{
public:
    ///
    static IrTypePointer* get(Type* dt);

    ///
    IrTypePointer* isPointer()      { return this; }

protected:
    ///
    IrTypePointer(Type* dt, llvm::Type *lt);
};

//////////////////////////////////////////////////////////////////////////////

/// IrType for static arrays
class IrTypeSArray : public IrType
{
public:
    ///
    static IrTypeSArray* get(Type* dt);

    ///
    IrTypeSArray* isSArray()  { return this; }

protected:
    ///
    IrTypeSArray(Type* dt);

    ///
    static llvm::Type* sarray2llvm(Type* t);
};

//////////////////////////////////////////////////////////////////////////////

/// IrType for dynamic arrays
class IrTypeArray : public IrType
{
public:
    ///
    static IrTypeArray* get(Type* dt);

    ///
    IrTypeArray* isArray()  { return this; }

protected:
    ///
    IrTypeArray(Type* dt, llvm::Type *lt);
};

//////////////////////////////////////////////////////////////////////////////

/// IrType for vectors
class IrTypeVector : public IrType
{
public:
    ///
    static IrTypeVector* get(Type* dt);

    ///
    IrTypeVector* isVector()    { return this; }

protected:
    ///
    IrTypeVector(Type* dt);

    static llvm::Type* vector2llvm(Type* dt);
};

#endif
