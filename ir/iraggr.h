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

//////////////////////////////////////////////////////////////////////////////

// represents a struct or class
// it is used during codegen to hold all the vital info we need
struct IrAggr
{
    /// Constructor.
    IrAggr(AggregateDeclaration* agg);

    //////////////////////////////////////////////////////////////////////////
    // public fields,
    // FIXME this is basically stuff I just haven't gotten around to yet.

    /// The D aggregate.
    AggregateDeclaration* aggrdecl;

    /// Aggregate D type.
    Type* type;

    /// true only for: align(1) struct S { ... }
    bool packed;

    //////////////////////////////////////////////////////////////////////////

    /// Create the __initZ symbol lazily.
    LLGlobalVariable* getInitSymbol();
    /// Builds the __initZ initializer constant lazily.
    LLConstant* getDefaultInit();

    /// Create the __vtblZ symbol lazily.
    LLGlobalVariable* getVtblSymbol();
    /// Builds the __vtblZ initializer constant lazily.
    LLConstant* getVtblInit();

    /// Create the __ClassZ/__InterfaceZ symbol lazily.
    LLGlobalVariable* getClassInfoSymbol();
    /// Builds the __ClassZ/__InterfaceZ initializer constant lazily.
    LLConstant* getClassInfoInit();

    /// Create the __interfaceInfos symbol lazily.
    LLGlobalVariable* getInterfaceArraySymbol();

    //////////////////////////////////////////////////////////////////////////

    /// Initialize interface.
    void initializeInterface();

    //////////////////////////////////////////////////////////////////////////

    typedef std::map<VarDeclaration*, llvm::Constant*> VarInitMap;

    /// Creates an initializer constant for the struct type with the given
    /// fields set to the provided constants. The remaining space (not
    /// explicitly specified fields, padding) is default-initialized.
    ///
    /// The optional initializerType parmeter can be used to specify the exact
    /// LLVM type to use for the initializer. If non-null and non-opaque, the
    /// type must exactly match the generated constant. This parameter is used
    /// mainly for supporting legacy code.
    ///
    /// Note that in the general case (if e.g. unions are involved), the
    /// returned type is not necessarily the same as getLLType().
    llvm::Constant* createInitializerConstant(
        const VarInitMap& explicitInitializers,
        llvm::StructType* initializerType = 0);

protected:
    /// Static default initializer global.
    LLGlobalVariable* init;
    /// Static default initializer constant.
    LLConstant* constInit;
    /// Static default initialier type.
    LLStructType* init_type;

    /// Vtbl global.
    LLGlobalVariable* vtbl;
    /// Vtbl initializer constant.
    LLConstant* constVtbl;

    /// ClassInfo global.
    LLGlobalVariable* classInfo;
    /// ClassInfo initializer constant.
    LLConstant* constClassInfo;

    /// Map for mapping ClassDeclaration* to LLVM GlobalVariable.
    typedef std::map<ClassDeclaration*, llvm::GlobalVariable*> ClassGlobalMap;

    /// Map from of interface vtbls implemented by this class.
    ClassGlobalMap interfaceVtblMap;

    /// Interface info array global.
    /// Basically: static object.Interface[num_interfaces]
    llvm::GlobalVariable* classInterfacesArray;

    /// std::vector of BaseClass*
    typedef std::vector<BaseClass*> BaseClassVector;

    /// Array of all interface vtbl implementations - in order - implemented
    /// by this class.
    /// Corresponds to the Interface instances needed to be output.
    BaseClassVector interfacesWithVtbls;

    //////////////////////////////////////////////////////////////////////////

    /// Returns vtbl for interface implementation, creates it if not already built.
    llvm::GlobalVariable* getInterfaceVtbl(
        BaseClass* b,
        bool new_inst,
        size_t interfaces_index);

    // FIXME make this a member instead
    friend LLConstant* DtoDefineClassInfo(ClassDeclaration* cd);

    /// Create the Interface[] interfaces ClassInfo field initializer.
    LLConstant* getClassInfoInterfaces();

private:
    /// Recursively adds all the initializers for the given aggregate and, in
    /// case of a class type, all its base classes.
    void addFieldInitializers(
        llvm::SmallVectorImpl<llvm::Constant*>& constants,
        const VarInitMap& explicitInitializers,
        AggregateDeclaration* decl,
        unsigned& offset,
        bool populateInterfacesWithVtbls);
};

//////////////////////////////////////////////////////////////////////////////

IrAggr *getIrAggr(AggregateDeclaration *decl, bool create = false);
bool isIrAggrCreated(AggregateDeclaration *decl);

#endif
