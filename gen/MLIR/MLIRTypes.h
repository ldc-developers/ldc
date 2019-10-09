//===-- ir/irtype.h - MLIRType base and primitive types -----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// The types derived from MLIRType are used to attach MLIR type information and
// other codegen metadata (e.g. for vtbl resolution) to frontend Types.
//
//===----------------------------------------------------------------------===//
/*
#pragma once

//#include "ir/irfuncty.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Builders.h"

//////////////////////////////////////////////////////////////////////////////

// forward declarations

namespace mlir {
class MLIRContext;
class Type;
class IntegerType;
}

class Type;

/*class MLIRTypeAggr;
class MLIRTypeArray;*/
//class MLIRTypeBasic;
/*class MLIRTypeClass;
class MLIRTypeDelegate;*/
//class MLIRTypeFunction;
/*class MLIRTypePointer;
class MLIRTypeSArray;
class MLIRTypeStruct;
class MLIRTypeVector;*/

//////////////////////////////////////////////////////////////////////////////

/// Code generation state/metadata for D types. The mapping from MLIRType to
/// Type is injective but not surjective.
///
/// Derived classes should be created using their static get() methods, which
/// makes sure that uniqueness is preserved in the face of forward references.
///
/// Note that the get() methods expect the MLIRType of the passed type/symbol not
/// to be set yet. Another option would be to just return the existing MLIRType
/// in such cases. This would bring the API more in line with the mlir::Type
/// get() functions. Currently all clients use the DtoType() wrapper anyway
/// instead of directly handling MLIRType instances, so keeping the assertions
/// allows us to check for any uniqueness violations that might have slipped
/// through.
// TODO: Implement the described changes (now that the forward reference
// handling logic seems to work correctly) and get rid of the "no-op" DtoType
// calls in MLIRAggr, ... that only exist for their side effect.
//class MLIRType {
//public:
//  virtual ~MLIRType() = default;

  ///
 /* virtual MLIRTypeAggr *isAggr() { return nullptr; }
  ///
  virtual MLIRTypeArray *isArray() { return nullptr; }*/
  ///
//  virtual MLIRTypeBasic *isBasic() { return nullptr; }
  ///
  /*virtual MLIRTypeClass *isClass() { return nullptr; }
  ///
  virtual MLIRTypeDelegate *isDelegate() { return nullptr; }*/
  ///
//  virtual MLIRTypeFunction *isFunction() { return nullptr; }
  ///
 /* virtual MLIRTypePointer *isPointer() { return nullptr; }
  ///
  virtual MLIRTypeSArray *isSArray() { return nullptr; }
  ///
  virtual MLIRTypeStruct *isStruct() { return nullptr; }
  ///
  virtual MLIRTypeVector *isVector() { return nullptr; }
*/
  ///
//  Type *getDType() { return dtype; }
  ///
 // virtual mlir::Type *getMLIRType() { return type; }

  ///
 // virtual MLIRFuncTy &getMLIRFuncTy();

  ///
  MLIRType(Type *dt, mlir::Type *lt, mlir::MLIRContext &mlirContext);

/*protected:
  ///
  Type *dtype = nullptr;

  /// MLIR type.
  mlir::Type *type = nullptr;
*/


//////////////////////////////////////////////////////////////////////////////
/*
/// MLIRType for basic D types.
class MLIRTypeBasic : public MLIRType {
public:
  ///
  static MLIRTypeBasic *get(Type *dt);

  ///
  MLIRTypeBasic *isBasic() override { return this; }

protected:
  ///
  explicit MLIRTypeBasic(Type *dt, mlir::MLIRContext &mlirContext);
  ///
  static mlir::Type *getComplexType(mlir::MLIRContext &ctx, mlir::Type *type);
  ///
  static mlir::Type *basic2mlir(Type *t, mlir::MLIRContext &mlirContext);
};

//////////////////////////////////////////////////////////////////////////////
/*
/// MLIRType from pointers.
class MLIRTypePointer : public MLIRType {
public:
  ///
  static MLIRTypePointer *get(Type *dt);

  ///
  MLIRTypePointer *isPointer() override { return this; }

protected:
  ///
  MLIRTypePointer(Type *dt, mlir::Type *lt);
};

//////////////////////////////////////////////////////////////////////////////

/// MLIRType for static arrays
class MLIRTypeSArray : public MLIRType {
public:
  ///
  static MLIRTypeSArray *get(Type *dt);

  ///
  MLIRTypeSArray *isSArray() override { return this; }

protected:
  ///
  MLIRTypeSArray(Type *dt, LLType *lt);
};

//////////////////////////////////////////////////////////////////////////////

/// MLIRType for dynamic arrays
class MLIRTypeArray : public MLIRType {
public:
  ///
  static MLIRTypeArray *get(Type *dt);

  ///
  MLIRTypeArray *isArray() override { return this; }

protected:
  ///
  MLIRTypeArray(Type *dt, mlir::Type *lt);
};

//////////////////////////////////////////////////////////////////////////////

/// MLIRType for vectors
class MLIRTypeVector : public MLIRType {
public:
  ///
  static MLIRTypeVector *get(Type *dt);

  ///
  MLIRTypeVector *isVector() override { return this; }

protected:
  ///
  explicit MLIRTypeVector(Type *dt, mlir::Type *lt);

  static mlir::Type *vector2mlir(Type *dt);
};
*/