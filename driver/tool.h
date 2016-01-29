//===-- driver/tool.h - External tool invocation helpers ---------*- C++
//-*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Functionaliy for invoking external tools executables, such as the system
// assembler, linker, ...
//
//===----------------------------------------------------------------------===//

#ifndef LDC_DRIVER_TOOL_H
#define LDC_DRIVER_TOOL_H

#include <vector>
#include <string>

int executeToolAndWait(const std::string &tool,
                       std::vector<std::string> const &args,
                       bool verbose = false);

#endif
