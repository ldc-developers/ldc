//===-- args.cpp ----------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "args.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/StringSaver.h"

#if LDC_WINDOWS_WMAIN
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

extern "C" {
// in driver/main.d:
int _Dmain(/*string[] args*/);

// in druntime:
int _d_run_main(int argc, const char **argv, int (*dMain)());
#if LDC_WINDOWS_WMAIN
int _d_wrun_main(int argc, const wchar_t **wargv, int (*dMain)());
#endif
}

namespace args {
static llvm::BumpPtrAllocator allocator;

void getCommandLineArguments(int argc, const CArgChar **argv,
    llvm::SmallVectorImpl<const char *> &result) {
#if LDC_WINDOWS_WMAIN
  // convert from UTF-16 to UTF-8
  result.reserve(argc);
  for (int i = 0; i < argc; ++i) {
    const wchar_t *warg = argv[i];
    const size_t wlen = wcslen(warg) + 1; // incl. terminating null
    const int len = WideCharToMultiByte(CP_UTF8, 0, warg, wlen, nullptr, 0,
                                        nullptr, nullptr);
    char *arg = allocator.Allocate<char>(len);
    WideCharToMultiByte(CP_UTF8, 0, warg, wlen, arg, len, nullptr, nullptr);
    result.push_back(arg);
  }
#else
#ifdef _WIN32
#pragma message ("Using current code page instead of UTF-8 for command-line args")
#endif
  result.insert(result.end(), argv, argv + argc);
#endif
}

void expandResponseFiles(llvm::SmallVectorImpl<const char *> &args) {
  llvm::StringSaver saver(allocator);
  llvm::cl::ExpandResponseFiles(saver,
#ifdef _WIN32
                          llvm::cl::TokenizeWindowsCommandLine
#else
                          llvm::cl::TokenizeGNUCommandLine
#endif
                          ,
                          args);
}

int forwardToDruntime(int argc, const CArgChar **argv) {
#if LDC_WINDOWS_WMAIN
  return _d_wrun_main(argc, argv, &_Dmain);
#else
  return _d_run_main(argc, argv, &_Dmain);
#endif
}

bool isRunArg(const char *arg) {
  return strcmp(arg, "-run") == 0 || strcmp(arg, "--run") == 0;
}
}
