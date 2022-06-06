//===-- ir/irtypeclass.h - IrType implementation for D classes --*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Provides the IrType subclass used to represent D classes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "ir/irtypeaggr.h"
#include "llvm/IR/DerivedTypes.h"

template <typename TYPE> struct Array;
class ClassDeclaration;
using FuncDeclarations = Array<class FuncDeclaration *>;
class TypeClass;

///
class IrTypeClass : public IrTypeAggr {
public:
  ///
  static IrTypeClass *get(ClassDeclaration *cd);

  ///
  IrTypeClass *isClass() override { return this; }

  ///
  llvm::Type *getLLType() override;

  /// Returns the actual storage type, i.e. without the indirection
  /// for the class reference.
  llvm::Type *getMemoryLLType();

  /// Returns the vtable type for this class.
  llvm::ArrayType *getVtblType() { return vtbl_type; }

  /// Get index to interface implementation.
  /// Returns the index of a specific interface implementation in this
  /// class or ~0 if not found.
  size_t getInterfaceIndex(ClassDeclaration *inter);

  /// Returns the number of interface implementations (vtables) in this
  /// class.
  unsigned getNumInterfaceVtbls() { return num_interface_vtbls; }

protected:
  ///
  explicit IrTypeClass(ClassDeclaration *cd);

  ///
  ClassDeclaration *cd = nullptr;
  ///
  TypeClass *tc = nullptr;

  /// Vtable type.
  llvm::ArrayType *vtbl_type = nullptr;

  /// Number of interface implementations (vtables) in this class.
  unsigned num_interface_vtbls = 0;

  /// std::map type mapping ClassDeclaration* to size_t.
  using ClassIndexMap = std::map<ClassDeclaration *, size_t>;

  /// Map for mapping the index of a specific interface implementation
  /// in this class to its ClassDeclaration.
  ClassIndexMap interfaceMap;

  //////////////////////////////////////////////////////////////////////////

  /// Adds the data members for the given class to the type builder, including
  /// those inherited from base classes/interfaces.
  void addClassData(AggrTypeBuilder &builder, ClassDeclaration *currCd);

  /// Adds the interface and all it's base interface to the interface
  /// to index map.
  void addInterfaceToMap(ClassDeclaration *inter, size_t index);
};
