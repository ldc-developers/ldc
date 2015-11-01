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

#ifndef LDC_GEN_WARNINGS_H
#define LDC_GEN_WARNINGS_H

#include "expression.h"
#include "mars.h"

void warnInvalidPrintfCall(Loc loc, Expression *arguments, size_t nargs);

#endif // LDC_GEN_WARNINGS_H
