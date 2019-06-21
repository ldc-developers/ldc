//===-- gen/warnings.h - LDC-specific warning handling ----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Functionality for emitting additional warnings during codegen.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "dmd/expression.h"

void warnInvalidPrintfCall(Loc loc, Expression *arguments, size_t nargs);
