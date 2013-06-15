//===-- gen/programs.h - External tool discovery ----------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Functions for discovering the external tools used for linking, etc.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_PROGRAMS_H
#define LDC_GEN_PROGRAMS_H

#if LDC_LLVM_VER >= 304

#include <string>

std::string getGcc();
std::string getArchiver();

// For Windows with MS tool chain
std::string getLink();
std::string getLib();

#else

#include "llvm/Support/Path.h"

llvm::sys::Path getGcc();
llvm::sys::Path getArchiver();

// For Windows with MS tool chain
llvm::sys::Path getLink();
llvm::sys::Path getLib();

#endif

#endif
