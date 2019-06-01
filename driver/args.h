//===-- driver/args.h - Command-line arguments management--------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/SmallVector.h"

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
}
