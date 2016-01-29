//===-- driver/exe_path.h - Executable path management ----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Stores the program's executable path and provides some helpers to generate
// derived paths.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_DRIVER_EXE_PATH_H
#define LDC_DRIVER_EXE_PATH_H

#include <string>

namespace exe_path {
void initialize(const char *arg0, void *mainAddress);

const std::string &getExePath();               // <baseDir>/bin/ldc2
std::string getBinDir();                       // <baseDir>/bin
std::string getBaseDir();                      // <baseDir>
std::string prependBinDir(const char *suffix); // <baseDir>/bin/<suffix>
}

#endif // LDC_DRIVER_EXE_PATH_H
