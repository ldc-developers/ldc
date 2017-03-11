//===-- driver/archiver.h - Creating static libs via LLVM--------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Provides an interface to LLVM built-in static lib generation via llvm-lib
// (MSVC targets) or llvm-ar (all other targets).
//
//===----------------------------------------------------------------------===//

#ifndef LDC_DRIVER_ARCHIVER_H
#define LDC_DRIVER_ARCHIVER_H

#if LDC_LLVM_VER >= 309
#include "llvm/ADT/ArrayRef.h"

namespace ldc {
int ar(llvm::ArrayRef<const char *> args);
int lib(llvm::ArrayRef<const char *> args);
}
#endif // LDC_LLVM_VER >= 309

#endif // !LDC_DRIVER_ARCHIVER_H
