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

#ifndef LDC_IR_IRAGGR_H
#define LDC_IR_IRAGGR_H

#include "llvm/ADT/SmallVector.h"
#include <map>
#include <vector>

// DMD forward declarations
class StructInitializer;

namespace llvm {
class Constant;
class StructType;
}

//////////////////////////////////////////////////////////////////////////////

// represents a struct or class
// it is used during codegen to hold all the vital info we need
struct IrAggr {
  /// Constructor.
  explicit IrAggr(AggregateDeclaration *aggr)
      : aggrdecl(aggr), type(aggr->type) {}

  //////////////////////////////////////////////////////////////////////////
  // public fields,
  // FIXME this is basically stuff I just haven't gotten around to yet.

  /// The D aggregate.
  AggregateDeclaration *aggrdecl = nullptr;

  /// Aggregate D type.
  Type *type = nullptr;

  //////////////////////////////////////////////////////////////////////////

  // Returns the static default initializer of a field.
  static llvm::Constant *getDefaultInitializer(VarDeclaration *field);

  //////////////////////////////////////////////////////////////////////////

  /// Create the __initZ symbol lazily.
  llvm::Constant *&getInitSymbol();
  /// Builds the __initZ initializer constant lazily.
  llvm::Constant *getDefaultInit();

  /// Create the __vtblZ symbol lazily.
  llvm::GlobalVariable *getVtblSymbol();
  /// Builds the __vtblZ initializer constant lazily.
  llvm::Constant *getVtblInit();

  /// Defines all interface vtbls.
  void defineInterfaceVtbls();

  /// Create the __ClassZ/__InterfaceZ symbol lazily.
  llvm::GlobalVariable *getClassInfoSymbol();
  /// Builds the __ClassZ/__InterfaceZ initializer constant lazily.
  llvm::Constant *getClassInfoInit();

  /// Create the __interfaceInfos symbol lazily.
  llvm::GlobalVariable *getInterfaceArraySymbol();

  //////////////////////////////////////////////////////////////////////////

  /// Initialize interface.
  void initializeInterface();

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

  /// Vtbl global.
  llvm::GlobalVariable *vtbl = nullptr;
  /// Vtbl initializer constant.
  llvm::Constant *constVtbl = nullptr;

  /// ClassInfo global.
  llvm::GlobalVariable *classInfo = nullptr;
  /// ClassInfo initializer constant.
  llvm::Constant *constClassInfo = nullptr;

  using ClassGlobalMap =
      std::map<std::pair<ClassDeclaration *, size_t>, llvm::GlobalVariable *>;

  /// Map from pairs of <interface vtbl,index> to global variable, implemented
  /// by this class. The same interface can appear multiple times, so index is
  /// another way to specify the thunk offset
  ClassGlobalMap interfaceVtblMap;

  /// Interface info array global.
  /// Basically: static object.Interface[num_interfaces]
  llvm::GlobalVariable *classInterfacesArray = nullptr;

  /// std::vector of BaseClass*
  using BaseClassVector = std::vector<BaseClass *>;

  /// Array of all interface vtbl implementations - in order - implemented
  /// by this class.
  /// Corresponds to the Interface instances needed to be output.
  BaseClassVector interfacesWithVtbls;

  //////////////////////////////////////////////////////////////////////////

  /// Returns the vtbl for an interface implementation.
  llvm::GlobalVariable *getInterfaceVtblSymbol(BaseClass *b,
                                               size_t interfaces_index);
  /// Defines the vtbl for an interface implementation.
  void defineInterfaceVtbl(BaseClass *b, bool new_inst,
                           size_t interfaces_index);

  // FIXME make this a member instead
  friend llvm::Constant *DtoDefineClassInfo(ClassDeclaration *cd);

  /// Create the Interface[] interfaces ClassInfo field initializer.
  llvm::Constant *getClassInfoInterfaces();

  /// Returns true, if the LLVM struct type for the aggregate is declared as
  /// packed
  bool isPacked() const;

private:
  llvm::StructType *llStructType = nullptr;

  llvm::StructType *getLLStructType();

  /// Recursively adds all the initializers for the given aggregate and, in
  /// case of a class type, all its base classes.
  void addFieldInitializers(llvm::SmallVectorImpl<llvm::Constant *> &constants,
                            const VarInitMap &explicitInitializers,
                            AggregateDeclaration *decl, unsigned &offset,
                            bool populateInterfacesWithVtbls);
};

//////////////////////////////////////////////////////////////////////////////

IrAggr *getIrAggr(AggregateDeclaration *decl, bool create = false);
bool isIrAggrCreated(AggregateDeclaration *decl);

#endif
