#ifndef LDC_GEN_METADATA_H
#define LDC_GEN_METADATA_H

#include "gen/llvm-version.h"

    // MDNode was moved into its own header, and contains Value*s
    #include "llvm/MDNode.h"
    typedef llvm::Value MDNodeField;
    
    // Use getNumElements() and getElement() to access elements.
    inline unsigned MD_GetNumElements(llvm::MDNode* N) {
        return N->getNumElements();
    }
    inline MDNodeField* MD_GetElement(llvm::MDNode* N, unsigned i) {
        return N->getElement(i);
    }

#define METADATA_LINKAGE_TYPE  llvm::GlobalValue::WeakODRLinkage

// *** Metadata for TypeInfo instances ***
#define TD_PREFIX "llvm.ldc.typeinfo."

/// The fields in the metadata node for a TypeInfo instance.
/// (Its name will be TD_PREFIX ~ <Name of TypeInfo global>)
enum TypeDataFields {
    TD_Confirm,     /// The TypeInfo this node is for.
    
    TD_Type,        /// A value of the LLVM type corresponding to this D type
    
    // Must be kept last:
    TD_NumFields    /// The number of fields in TypeInfo metadata
};


// *** Metadata for ClassInfo instances ***
#define CD_PREFIX "llvm.ldc.classinfo."

/// The fields in the metadata node for a ClassInfo instance.
/// (Its name will be CD_PREFIX ~ <Name of ClassInfo global>)
enum ClassDataFields {
    CD_BodyType,    /// A value of the LLVM type corresponding to the class body.
    CD_Finalize,    /// True if this class (or a base class) has a destructor.
    CD_CustomDelete,/// True if this class has an overridden delete operator.
    
    // Must be kept last
    CD_NumFields    /// The number of fields in ClassInfo metadata
};

#endif
