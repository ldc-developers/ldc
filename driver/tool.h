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

#pragma once

#include <vector>
#include <string>

#include "llvm/Support/CommandLine.h"

struct Loc;

namespace opts {
extern llvm::cl::opt<std::string> linker;
}

std::string getGcc(std::vector<std::string> &additional_args,
		   const char *fallback = "cc");
void appendTargetArgsForGcc(std::vector<std::string> &args);

std::string getProgram(const char *fallbackName,
                       const llvm::cl::opt<std::string> *opt = nullptr,
                       const char *envVar = nullptr);

void createDirectoryForFileOrFail(llvm::StringRef fileName);

// NB: `args` must outlive the returned vector!
std::vector<const char *> getFullArgs(const char *tool,
                                      const std::vector<std::string> &args,
                                      bool printVerbose);

int executeToolAndWait(const Loc &loc, const std::string &tool,
                       const std::vector<std::string> &args,
                       bool verbose = false);

#ifdef _WIN32

namespace windows {
// Returns true if a usable MSVC installation is available.
bool isMsvcAvailable();

struct MsvcEnvironmentScope {
  // Tries to set up the MSVC environment variables for the current process and
  // returns true if successful. The original environment is restored on
  // destruction.
  bool setup(bool forPreprocessingOnly = false);

  ~MsvcEnvironmentScope();

private:
  // for each changed env var: name & original value
  std::vector<std::pair<std::wstring, wchar_t *>> rollback;
};
}

#endif
