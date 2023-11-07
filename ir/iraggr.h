//===-- ir/iraggr.h - Codegen state for D aggregates ------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Represents the state of a D aggregate (struct/class) on its way through
// codegen, also managing the associated init and RTTI symbols.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "dmd/aggregate.h"
#include "llvm/ADT/SmallVector.h"
#include <map>
#include <vector>

class StructInitializer;

namespace llvm {
class Constant;
class DIType;
class GlobalVariable;
class StructType;
}

//////////////////////////////////////////////////////////////////////////////

/// Represents a struct or class/interface.
/// It is used during codegen to hold all the vital info we need.
class IrAggr {
public:
  //////////////////////////////////////////////////////////////////////////
  // public fields,
  // FIXME this is basically stuff I just haven't gotten around to yet.

  /// The D aggregate.
  AggregateDeclaration *aggrdecl = nullptr;

  /// Aggregate D type.
  Type *type = nullptr;

  /// Composite type debug description. This is not only to cache, but also
  /// used for resolving forward references.
  llvm::DIType *diCompositeType = nullptr;

  //////////////////////////////////////////////////////////////////////////

  /// Returns the static default initializer of a field.
  static llvm::Constant *getDefaultInitializer(VarDeclaration *field);

  //////////////////////////////////////////////////////////////////////////

  virtual ~IrAggr() = default;

  /// Creates the __initZ symbol lazily.
  llvm::Constant *getInitSymbol(bool define = false);
  /// Builds the __initZ initializer constant lazily.
  llvm::Constant *getDefaultInit();
  /// Return the LLVM type of this Aggregate (w/o the reference for classes)
  llvm::StructType *getLLStructType();

  /// Whether to suppress the TypeInfo definition for the aggregate via
  /// `-betterC` / `-fno-rtti`, no `object.TypeInfo`, or
  /// `pragma(LDC_no_typeinfo)`.
  bool suppressTypeInfo() const;

  //////////////////////////////////////////////////////////////////////////

  using VarInitMap = std::map<VarDeclaration *, llvm::Constant *>;

  /// Creates an initializer constant for the struct type with the given
  /// fields set to the provided constants. The remaining space (not
  /// explicitly specified fields, padding) is default-initialized.
  ///
  /// Note that in the general case (if e.g. unions are involved), the
  /// returned type is not necessarily the same as getLLType().
  llvm::Constant *
  createInitializerConstant(const VarInitMap &explicitInitializers);

protected:
  /// Static default initializer global.
  llvm::Constant *init = nullptr;
  /// Static default initializer constant.
  llvm::Constant *constInit = nullptr;

  /// TypeInfo global.
  llvm::GlobalVariable *typeInfo = nullptr;
  /// TypeInfo initializer constant.
  llvm::Constant *constTypeInfo = nullptr;

  explicit IrAggr(AggregateDeclaration *aggr)
      : aggrdecl(aggr), type(aggr->type) {}

  // Use dllimport for *declared* init symbol, TypeInfo and vtable?
  bool useDLLImport() const;

private:
  llvm::StructType *llStructType = nullptr;

  /// Recursively adds all the initializers for the given aggregate and, in
  /// case of a class type, all its base classes.
  void addFieldInitializers(llvm::SmallVectorImpl<llvm::Constant *> &constants,
                            const VarInitMap &explicitInitializers,
                            AggregateDeclaration *decl, unsigned &offset,
                            unsigned &interfaceVtblIndex, bool &isPacked);
};

/// Represents a struct.
class IrStruct : public IrAggr {
public:
  explicit IrStruct(StructDeclaration *sd) : IrAggr(sd) {}

  /// Creates the TypeInfo_Struct symbol lazily.
  llvm::GlobalVariable *getTypeInfoSymbol(bool define = false);

private:
  /// Builds the TypeInfo_Struct initializer constant lazily.
  llvm::Constant *getTypeInfoInit();
};

/// Represents a class/interface.
class IrClass : public IrAggr {
public:
  explicit IrClass(ClassDeclaration *cd);

  /// Creates the __ClassZ/__InterfaceZ symbol lazily.
  llvm::GlobalVariable *getClassInfoSymbol(bool define = false);

  /// Creates the __vtblZ symbol lazily.
  llvm::GlobalVariable *getVtblSymbol(bool define = false);

  /// Defines all interface vtbls.
  void defineInterfaceVtbls();

private:
  /// Vtbl global.
  llvm::GlobalVariable *vtbl = nullptr;
  /// Vtbl initializer constant.
  llvm::Constant *constVtbl = nullptr;

  /// Map from pairs of <interface vtbl,index> to global variable, implemented
  /// by this class. The same interface can appear multiple times, so index is
  /// another way to specify the thunk offset
  std::map<std::pair<ClassDeclaration *, size_t>, llvm::GlobalVariable *>
      interfaceVtblMap;

  /// Interface info array global.
  /// Basically: static object.Interface[num_interfaces]
  llvm::GlobalVariable *classInterfacesArray = nullptr;

  /// Array of all interface vtbl implementations - in order - implemented
  /// by this class.
  /// Corresponds to the Interface instances needed to be output.
  std::vector<BaseClass *> interfacesWithVtbls;

  void addInterfaceVtbls(ClassDeclaration *cd);

  /// Builds the __ClassZ/__InterfaceZ initializer constant lazily.
  llvm::Constant *getClassInfoInit();

  /// Builds the __vtblZ initializer constant lazily.
  llvm::Constant *getVtblInit();

  /// Returns the vtbl for an interface implementation.
  llvm::GlobalVariable *getInterfaceVtblSymbol(BaseClass *b,
                                               size_t interfaces_index,
                                               bool define = false);
  /// Defines the vtbl for an interface implementation.
  llvm::Constant *getInterfaceVtblInit(BaseClass *b, size_t interfaces_index);

  /// Creates the __interfaceInfos symbol lazily.
  llvm::GlobalVariable *getInterfaceArraySymbol();

  /// Create the Interface[] interfaces ClassInfo field initializer.
  llvm::Constant *getClassInfoInterfaces();

  // FIXME: IrAggr::createInitializerConstant() needs full access
  friend class IrAggr;
};

//////////////////////////////////////////////////////////////////////////////

IrAggr *getIrAggr(AggregateDeclaration *decl, bool create = false);
IrStruct *getIrAggr(StructDeclaration *sd, bool create = false);
IrClass *getIrAggr(ClassDeclaration *cd, bool create = false);

bool isIrAggrCreated(AggregateDeclaration *decl);
