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

#ifndef __LDC_IR_IRTYPECLASS_H__
#define __LDC_IR_IRTYPECLASS_H__

#include "ir/irtypeaggr.h"
#if LDC_LLVM_VER >= 303
#include "llvm/IR/DerivedTypes.h"
#else
#include "llvm/DerivedTypes.h"
#endif

template <typename TYPE> struct Array;
typedef Array<class FuncDeclaration *> FuncDeclarations;

///
class IrTypeClass : public IrTypeAggr
{
public:
    ///
    static IrTypeClass* get(ClassDeclaration* cd);

    ///
    virtual IrTypeClass* isClass()      { return this; }

    ///
    llvm::Type* getLLType();

    /// Returns the actual storage type, i.e. without the indirection
    /// for the class reference.
    llvm::Type* getMemoryLLType();

    /// Returns the vtable type for this class.
    llvm::Type* getVtbl()         { return vtbl_type; }

    /// Get index to interface implementation.
    /// Returns the index of a specific interface implementation in this
    /// class or ~0 if not found.
    size_t getInterfaceIndex(ClassDeclaration* inter);

    /// Returns the total number of pointers in the vtable.
    unsigned getVtblSize()              { return vtbl_size; }

    /// Returns the number of interface implementations (vtables) in this
    /// class.
    unsigned getNumInterfaceVtbls()     { return num_interface_vtbls; }

protected:
    ///
    IrTypeClass(ClassDeclaration* cd);

    ///
    ClassDeclaration* cd;
    ///
    TypeClass* tc;

    /// Vtable type.
    llvm::StructType *vtbl_type;

    /// Number of pointers in vtable.
    unsigned vtbl_size;

    /// Number of interface implementations (vtables) in this class.
    unsigned num_interface_vtbls;

    /// std::map type mapping ClassDeclaration* to size_t.
    typedef std::map<ClassDeclaration*, size_t> ClassIndexMap;

    /// Map for mapping the index of a specific interface implementation
    /// in this class to its ClassDeclaration.
    ClassIndexMap interfaceMap;

    //////////////////////////////////////////////////////////////////////////

    /// Builds a vtable type given the type of the first entry and an array
    /// of all entries.
   std::vector<llvm::Type*> buildVtblType(Type* first, FuncDeclarations* vtbl_array);

    ///
    void addBaseClassData(AggrTypeBuilder &builder, ClassDeclaration *base);

    /// Adds the interface and all it's base interface to the interface
    /// to index map.
    void addInterfaceToMap(ClassDeclaration* inter, size_t index);
};

#endif
