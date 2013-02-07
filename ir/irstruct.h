//===-- ir/irstruct.h - Codegen state for D aggregates ----------*- C++ -*-===//
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

#ifndef LDC_IR_IRSTRUCT_H
#define LDC_IR_IRSTRUCT_H

#include "ir/ir.h"
#include <map>
#include <vector>

// DMD forward declarations
struct StructInitializer;

//////////////////////////////////////////////////////////////////////////////

// represents a struct or class
// it is used during codegen to hold all the vital info we need
struct IrStruct : IrBase
{
    /// Constructor.
    IrStruct(AggregateDeclaration* agg);

    //////////////////////////////////////////////////////////////////////////
    // public fields,
    // FIXME this is basically stuff I just haven't gotten around to yet.

    /// The D aggregate.
    AggregateDeclaration* aggrdecl;

    /// Aggregate D type.
    Type* type;

    /// true only for: align(1) struct S { ... }
    bool packed;

    /// Composite type debug description.
    llvm::DIType diCompositeType;

    //////////////////////////////////////////////////////////////////////////

    /// Create the __initZ symbol lazily.
    LLGlobalVariable* getInitSymbol();
    /// Builds the __initZ initializer constant lazily.
    LLConstant* getDefaultInit();

    /// Create the __vtblZ symbol lazily.
    LLGlobalVariable* getVtblSymbol();
    /// Builds the __vtblZ initializer constant lazily.
    LLConstant* getVtblInit();

    /// Create the __ClassZ symbol lazily.
    LLGlobalVariable* getClassInfoSymbol();
    /// Builds the __ClassZ initializer constant lazily.
    LLConstant* getClassInfoInit();

    /// Create the __interfaceInfos symbol lazily.
    LLGlobalVariable* getInterfaceArraySymbol();

    /// Creates a StructInitializer constant.
    LLConstant* createStructInitializer(StructInitializer* si);

    //////////////////////////////////////////////////////////////////////////

    /// Initialize interface.
    void initializeInterface();

    //////////////////////////////////////////////////////////////////////////
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

    /// Create static default initializer for struct.
    std::vector<llvm::Constant*> createStructDefaultInitializer();

    /// Create static default initializer for class.
    std::vector<llvm::Constant*> createClassDefaultInitializer();

    /// Returns vtbl for interface implementation, creates it if not already built.
    llvm::GlobalVariable* getInterfaceVtbl(
        BaseClass* b,
        bool new_inst,
        size_t interfaces_index);

    /// Add base class data to initializer list.
    /// Also creates the IrField instance for each data field.
    void addBaseClassInits(
        std::vector<llvm::Constant*>& constants,
        ClassDeclaration* base,
        size_t& offset,
        size_t& field_index);

    // FIXME make this a member instead
    friend LLConstant* DtoDefineClassInfo(ClassDeclaration* cd);

    /// Create the Interface[] interfaces ClassInfo field initializer.
    LLConstant* getClassInfoInterfaces();
};

//////////////////////////////////////////////////////////////////////////////

#endif
