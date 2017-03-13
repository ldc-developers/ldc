//===-- driver/tool.h - External tool invocation helpers --------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Functionality for invoking external tools executables, such as the system
// assembler, linker, ...
//
//===----------------------------------------------------------------------===//

#ifndef LDC_DRIVER_TOOL_H
#define LDC_DRIVER_TOOL_H

#include <vector>
#include <string>

std::string getGcc();
std::string getArchiver();

int executeToolAndWait(const std::string &tool,
                       std::vector<std::string> const &args,
                       bool verbose = false);

#ifdef _WIN32

namespace windows {
// Tries to set up the MSVC environment variables and returns true if
// successful.
bool setupMsvcEnvironment();
}

#endif

#endif
