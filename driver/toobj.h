//===-- driver/toobj.h - Object file emission -------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Handles emission of "finished" LLVM modules to on-disk object files.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace llvm {
class Module;
}

void writeModule(llvm::Module *m, const char *filename);
