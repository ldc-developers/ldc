//===-- logger.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/logger.h"

#include "dmd/errors.h"
#include "gen/irstate.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_os_ostream.h"
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

void Stream::writeType(std::ostream &OS, const llvm::Type &Ty) {
  llvm::raw_os_ostream raw(OS);
  Ty.print(raw);
}

void Stream::writeValue(std::ostream &OS, const llvm::Value &V) {
  // Constants don't always get their types pretty-printed.
  // (Only treat non-global constants like this, so that e.g. global variables
  // still get their initializers printed)
  llvm::raw_os_ostream raw(OS);
  if (llvm::isa<llvm::Constant>(V) && !llvm::isa<llvm::GlobalValue>(V)) {
    V.printAsOperand(raw, true, &gIR->module);
  } else {
    V.print(raw);
  }
}

// This variable is pulled out of the Logger namespace to work around D bug 15576.
// https://issues.dlang.org/show_bug.cgi?id=15576
bool _Logger_enabled;

namespace Logger {
static std::string indent_str;

static llvm::cl::opt<bool, true>
    enabledopt("vv", llvm::cl::desc("Print front-end/glue code debug log"),
               llvm::cl::location(_Logger_enabled), llvm::cl::ZeroOrMore);

void indent() {
  if (_Logger_enabled) {
    indent_str += "* ";
  }
}

void undent() {
  if (_Logger_enabled) {
    assert(!indent_str.empty());
    indent_str.resize(indent_str.size() - 2);
  }
}

Stream cout() {
  if (_Logger_enabled) {
    return Stream(std::cout << indent_str);
  }
  return Stream(nullptr);
}

#if defined(_MSC_VER)
static inline void search_and_replace(std::string &str, const std::string &what,
                                      const std::string &replacement) {
  assert(!what.empty());
  size_t pos = str.find(what);
  while (pos != std::string::npos) {
    str.replace(pos, what.size(), replacement);
    pos = str.find(what, pos + replacement.size());
  }
}

#define WORKAROUND_C99_SPECIFIERS_BUG(f)                                       \
  std::string tmp = f;                                                         \
  search_and_replace(tmp, std::string("%z"), std::string("%I"));               \
  f = tmp.c_str();
#else
#define WORKAROUND_C99_SPECIFIERS_BUG(f)
#endif

void printIndentation() { printf("%s", indent_str.c_str()); }

void println(const char *fmt, ...) {
  if (_Logger_enabled) {
    printf("%s", indent_str.c_str());
    va_list va;
    va_start(va, fmt);
    WORKAROUND_C99_SPECIFIERS_BUG(fmt);
    vprintf(fmt, va);
    va_end(va);
    printf("\n");
  }
}
void print(const char *fmt, ...) {
  if (_Logger_enabled) {
    printf("%s", indent_str.c_str());
    va_list va;
    va_start(va, fmt);
    WORKAROUND_C99_SPECIFIERS_BUG(fmt);
    vprintf(fmt, va);
    va_end(va);
  }
}
}
