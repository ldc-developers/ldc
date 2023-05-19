//===-- driver/args.h - Command-line & environment variables ----*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Program.h"

namespace args {

// On Windows, the host LDC druntime features _d_wrun_main with UTF-16 command-
// line arguments support starting with v2.087.
#if defined(_WIN32) && LDC_HOST_FE_VER >= 2087
#define LDC_WINDOWS_WMAIN 1
using CArgChar = wchar_t;
#else
using CArgChar = char;
#endif

// Converts the native command-line to a vector of UTF-8 arguments.
void getCommandLineArguments(int argc, const CArgChar **argv,
                             llvm::SmallVectorImpl<const char *> &result);

// Expands any response files (@file), in-place.
void expandResponseFiles(llvm::SmallVectorImpl<const char *> &args);

// Calls _d_run_main with the specified arguments, initializing druntime and
// continuing with _Dmain in driver/main.d.
int forwardToDruntime(int argc, const CArgChar **argv);

// Returns true if the specified arg is either `-run` or `--run`.
bool isRunArg(const char *arg);

// Executes a command line and returns its exit code.
// Optionally uses a response file to overcome cmdline length limitations.
int executeAndWait(std::vector<const char *> fullArgs,
                   llvm::sys::WindowsEncodingMethod responseFileEncoding,
                   std::string *errorMsg = nullptr);
} // namespace args

////////////////////////////////////////////////////////////////////////////////

namespace env {

// Indicates whether the specified environment variable is set.
bool has(const char *name);
#ifdef _WIN32
bool has(const wchar_t *wname);
#endif

// Returns the value of the specified environment variable (in UTF-8).
std::string get(const char *name);
} // namespace env
