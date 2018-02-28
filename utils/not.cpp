//===- not.cpp - The 'not' testing tool -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Usage:
//   not cmd
//     Will return true if cmd doesn't crash and returns false.
//   not --crash cmd
//     Will return true if cmd crashes (e.g. for testing crash reporting).

#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

int main(int argc, const char **argv) {
  bool ExpectCrash = false;

  ++argv;
  --argc;

  if (argc > 0 && StringRef(argv[0]) == "--crash") {
    ++argv;
    --argc;
    ExpectCrash = true;
  }

  if (argc == 0)
    return 1;

#if LDC_LLVM_VER <= 305
  auto Program = sys::FindProgramByName(argv[0]);
#else
  auto Program = sys::findProgramByName(argv[0]);
  if (!Program) {
    errs() << "Error: Unable to find `" << argv[0]
           << "' in PATH: " << Program.getError().message() << "\n";
    return 1;
  }
#endif

  std::string ErrMsg;
#if LDC_LLVM_VER <= 305
  int Result = sys::ExecuteAndWait(Program, argv, nullptr, nullptr, 0, 0,
                                   &ErrMsg);
#else
  int Result = sys::ExecuteAndWait(*Program, argv, nullptr,
#if LDC_LLVM_VER >= 600
                                   {},
#else
                                   nullptr,
#endif
                                   0, 0, &ErrMsg);
#endif
#ifdef _WIN32
  // Handle abort() in msvcrt -- It has exit code as 3.  abort(), aka
  // unreachable, should be recognized as a crash.  However, some binaries use
  // exit code 3 on non-crash failure paths, so only do this if we expect a
  // crash.
  if (ExpectCrash && Result == 3)
    Result = -3;
#endif
  if (Result < 0) {
    errs() << "Error: " << ErrMsg << "\n";
    if (ExpectCrash)
      return 0;
    return 1;
  }

  if (ExpectCrash)
    return 1;

  return Result == 0;
}
