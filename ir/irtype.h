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

#pragma once

#include "ir/irfuncty.h"

//////////////////////////////////////////////////////////////////////////////

// forward declarations

namespace llvm {
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
///
/// Note that the get() methods expect the IrType of the passed type/symbol not
/// to be set yet. Another option would be to just return the existing IrType
/// in such cases. This would bring the API more in line with the llvm::Type
/// get() functions. Currently all clients use the DtoType() wrapper anyway
/// instead of directly handling IrType instances, so keeping the assertions
/// allows us to check for any uniqueness violations that might have slipped
/// through.
// TODO: Implement the described changes (now that the forward reference
// handling logic seems to work correctly) and get rid of the "no-op" DtoType
// calls in IrAggr, ... that only exist for their side effect.
class IrType {
public:
  virtual ~IrType() = default;

  ///
  virtual IrTypeAggr *isAggr() { return nullptr; }
  ///
  virtual IrTypeArray *isArray() { return nullptr; }
  ///
  virtual IrTypeBasic *isBasic() { return nullptr; }
  ///
  virtual IrTypeClass *isClass() { return nullptr; }
  ///
  virtual IrTypeDelegate *isDelegate() { return nullptr; }
  ///
  virtual IrTypeFunction *isFunction() { return nullptr; }
  ///
  virtual IrTypePointer *isPointer() { return nullptr; }
  ///
  virtual IrTypeSArray *isSArray() { return nullptr; }
  ///
  virtual IrTypeStruct *isStruct() { return nullptr; }
  ///
  virtual IrTypeVector *isVector() { return nullptr; }

  ///
  Type *getDType() { return dtype; }
  ///
  virtual llvm::Type *getLLType() { return type; }

  ///
  virtual IrFuncTy &getIrFuncTy();

protected:
  ///
  IrType(Type *dt, llvm::Type *lt);

  ///
  Type *dtype = nullptr;

  /// LLVM type.
  llvm::Type *type = nullptr;
};

//////////////////////////////////////////////////////////////////////////////

/// IrType for basic D types.
class IrTypeBasic : public IrType {
public:
  ///
  static IrTypeBasic *get(Type *dt);

  ///
  IrTypeBasic *isBasic() override { return this; }

protected:
  ///
  explicit IrTypeBasic(Type *dt);
  ///
  static llvm::Type *getComplexType(llvm::LLVMContext &ctx, llvm::Type *type);
  ///
  static llvm::Type *basic2llvm(Type *t);
};

//////////////////////////////////////////////////////////////////////////////

/// IrType from pointers.
class IrTypePointer : public IrType {
public:
  ///
  static IrTypePointer *get(Type *dt);

  ///
  IrTypePointer *isPointer() override { return this; }

protected:
  ///
  IrTypePointer(Type *dt, llvm::Type *lt);
};

//////////////////////////////////////////////////////////////////////////////

/// IrType for static arrays
class IrTypeSArray : public IrType {
public:
  ///
  static IrTypeSArray *get(Type *dt);

  ///
  IrTypeSArray *isSArray() override { return this; }

protected:
  ///
  IrTypeSArray(Type *dt, LLType *lt);
};

//////////////////////////////////////////////////////////////////////////////

/// IrType for dynamic arrays
class IrTypeArray : public IrType {
public:
  ///
  static IrTypeArray *get(Type *dt);

  ///
  IrTypeArray *isArray() override { return this; }

protected:
  ///
  IrTypeArray(Type *dt, llvm::Type *lt);
};

//////////////////////////////////////////////////////////////////////////////

/// IrType for vectors
class IrTypeVector : public IrType {
public:
  ///
  static IrTypeVector *get(Type *dt);

  ///
  IrTypeVector *isVector() override { return this; }

protected:
  ///
  explicit IrTypeVector(Type *dt, llvm::Type *lt);
};

//////////////////////////////////////////////////////////////////////////////

/// Returns a reference to the IrType* associated with the specified D type.
IrType *&getIrType(Type *t, bool create = false);
