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

#ifndef __LDC_GEN_WARNINGS_H__
#define __LDC_GEN_WARNINGS_H__

void warnInvalidPrintfCall(Loc loc, Expression* arguments, size_t nargs);

#endif // __LDC_GEN_WARNINGS_H__
