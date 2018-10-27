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

#pragma once

#include "llvm/ADT/Twine.h"

#include <string>

namespace exe_path {

void initialize(const char *arg0);

const std::string &getExePath();               // <baseDir>/bin/ldc2
std::string getBinDir();                       // <baseDir>/bin
std::string getBaseDir();                      // <baseDir>
std::string getLibDir();                       // <baseDir>/lib
std::string prependBinDir(const llvm::Twine &suffix); // <baseDir>/bin/<suffix>
std::string prependLibDir(const llvm::Twine &suffix); // <baseDir>/lib/<suffix>
}
