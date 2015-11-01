//===-- gen/metadata.h - LDC-specific LLVM metadata definitions -*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Defines the types of LLVM metadata used for D-specific optimization passes.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_METADATA_H
#define LDC_GEN_METADATA_H

// MDNode was moved into its own header, and contains Value*s
#include "llvm/IR/Metadata.h"
typedef llvm::Value MDNodeField;

#define METADATA_LINKAGE_TYPE  llvm::GlobalValue::WeakODRLinkage

// *** Metadata for TypeInfo instances ***
#define TD_PREFIX "llvm.ldc.typeinfo."

/// The fields in the metadata node for a TypeInfo instance.
/// (Its name will be TD_PREFIX ~ <Name of TypeInfo global>)
enum TypeDataFields {
    TD_TypeInfo,    /// A reference toe the TypeInfo global this node is for.

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
