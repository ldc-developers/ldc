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

#pragma once

#include "gen/llvm.h"

// Make it easier to test new linkage types

#define TYPEINFO_LINKAGE_TYPE LLGlobalValue::LinkOnceODRLinkage

extern LLGlobalValue::LinkageTypes templateLinkage;
