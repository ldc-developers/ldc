//===-- gen/linkage.h - Common linkage types --------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Linkage types used for certain constructs (templates, TypeInfo).
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_LINKAGE_H
#define LDC_GEN_LINKAGE_H

#include "gen/llvm.h"

// Make it easier to test new linkage types

#define TYPEINFO_LINKAGE_TYPE LLGlobalValue::LinkOnceODRLinkage
// The One-Definition-Rule shouldn't matter for debug info, right?
#define DEBUGINFO_LINKONCE_LINKAGE_TYPE LLGlobalValue::LinkOnceAnyLinkage

extern LLGlobalValue::LinkageTypes templateLinkage;

#endif
