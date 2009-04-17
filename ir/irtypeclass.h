#ifndef __LDC_IR_IRTYPECLASS_H__
#define __LDC_IR_IRTYPECLASS_H__

#include "ir/irtypestruct.h"

///
class IrTypeClass : public IrTypeAggr
{
public:
    ///
    IrTypeClass(ClassDeclaration* cd);

    ///
    virtual IrTypeClass* isClass()      { return this; }

    ///
    const llvm::Type* buildType();

    ///
    const llvm::Type* get();

    /// Returns the vtable type for this class.
    const llvm::Type* getVtbl()         { return vtbl_pa.get(); }

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
    ClassDeclaration* cd;
    ///
    TypeClass* tc;

    /// Type holder for the vtable type.
    llvm::PATypeHolder vtbl_pa;

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
    const llvm::Type* buildVtblType(Type* first, Array* vtbl_array);

    ///
    void addBaseClassData(
        std::vector<const llvm::Type*>& defaultTypes,
        ClassDeclaration* base,
        size_t& offset,
        size_t& field_index);

    /// Adds the interface and all it's base interface to the interface
    /// to index map.
    void addInterfaceToMap(ClassDeclaration* inter, size_t index);
};

#endif
