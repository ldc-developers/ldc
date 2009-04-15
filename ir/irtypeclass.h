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
    const llvm::Type* getVtbl()         { return vtbl_pa.get(); }

    ///
    const llvm::Type* get();

    /// Get index to interface implementation.
    /// Returns the index of a specific interface implementation in this
    /// class or ~0 if not found.
    size_t getInterfaceIndex(ClassDeclaration* inter);

    /// Returns the total number of pointers in the vtable.
    unsigned getVtblSize()              { return vtbl_size; }

protected:
    ///
    ClassDeclaration* cd;
    ///
    TypeClass* tc;

    ///
    llvm::PATypeHolder vtbl_pa;

    /// Number of pointers in vtable.
    unsigned vtbl_size;

    /// std::map type mapping ClassDeclaration* to size_t.
    typedef std::map<ClassDeclaration*, size_t> ClassIndexMap;

    /// Map for mapping the index of a specific interface implementation
    /// in this class to it's ClassDeclaration*.
    ClassIndexMap interfaceMap;

    //////////////////////////////////////////////////////////////////////////

    ///
    const llvm::Type* buildVtblType(Type* first, Array* vtbl_array);

    ///
    void addBaseClassData(
        std::vector<const llvm::Type*>& defaultTypes,
        ClassDeclaration* base,
        size_t& offset,
        size_t& field_index);
};

#endif
