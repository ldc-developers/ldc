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

#include <string>

std::string getProgram(const char *name, const char *envVar = 0);

std::string getGcc();
std::string getArchiver();

#endif
