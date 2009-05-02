#ifndef LDC_GEN_METADATA_H
#define LDC_GEN_METADATA_H

#include "gen/llvm-version.h"

#if LLVM_REV >= 68420
// Yay, we have metadata!

#define USE_METADATA
#define METADATA_LINKAGE_TYPE  llvm::GlobalValue::WeakODRLinkage

// *** Metadata for TypeInfo instances ***
#define TD_PREFIX "llvm.ldc.typeinfo."

/// The fields in the metadata node for a TypeInfo instance.
/// (Its name will be TD_PREFIX ~ <Name of TypeInfo global>)
enum TypeDataFields {
    TD_Confirm,     /// The TypeInfo this node is for
    TD_Type,        /// A value of the LLVM type corresponding to this D type
    
    // Must be kept last:
    TD_NumFields    /// The number of fields in TypeInfo metadata
};

#endif

#endif
