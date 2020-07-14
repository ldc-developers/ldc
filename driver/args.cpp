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

#include <cstdlib>

#ifdef _WIN32
#include "llvm/Support/ConvertUTF.h"
#endif

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
} // namespace args

namespace env {
#ifdef _WIN32
static wchar_t *wget(const char *name) {
  llvm::SmallVector<wchar_t, 32> wname;
  llvm::sys::windows::UTF8ToUTF16(name, wname);
  wname.push_back(0);
  return _wgetenv(wname.data());
}
#endif

bool has(const char *name) {
#ifdef _WIN32
  return wget(name) != nullptr;
#else
  return getenv(name) != nullptr;
#endif
}

#ifdef _WIN32
bool has(const wchar_t *wname) { return _wgetenv(wname) != nullptr; }
#endif

std::string get(const char *name) {
#ifdef _WIN32
#if LDC_LLVM_VER >= 400
  using llvm::UTF16;
#endif
  const wchar_t *wvalue = wget(name);
  std::string value;
  if (wvalue) {
    llvm::convertUTF16ToUTF8String(
        {reinterpret_cast<const UTF16 *>(wvalue), wcslen(wvalue)}, value);
  }
  return value;
#else
  const char *value = getenv(name);
  return value ? value : "";
#endif
}
}
