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

#include "llvm/Support/CommandLine.h"

namespace opts {
extern llvm::cl::opt<std::string> linker;
}

std::string getGcc();
void appendTargetArgsForGcc(std::vector<std::string> &args);

std::string getProgram(const char *name,
                       const llvm::cl::opt<std::string> *opt = nullptr,
                       const char *envVar = nullptr);

void createDirectoryForFileOrFail(llvm::StringRef fileName);

// NB: `args` must outlive the returned vector!
std::vector<const char *> getFullArgs(const char *tool,
                                      const std::vector<std::string> &args,
                                      bool printVerbose);

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
