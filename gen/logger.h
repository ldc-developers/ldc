//===-- gen/logger.h - Codegen debug logging --------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Defines a common interface for logging debug information during code
// generation.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <iosfwd>
#include <iostream>

namespace llvm {
class Type;
class Value;
}

#ifndef IS_PRINTF
#ifdef __GNUC__
#define IS_PRINTF(FMTARG)                                                      \
  __attribute((__format__(__printf__, (FMTARG), (FMTARG) + 1)))
#else
#define IS_PRINTF(FMTARG)
#endif
#endif

struct Loc;

class Stream {
  std::ostream *OS;

public:
  Stream() : OS(nullptr) {}
  explicit Stream(std::ostream *S) : OS(S) {}
  explicit Stream(std::ostream &S) : OS(&S) {}

  /*
  Stream operator << (std::ios_base &(*Func)(std::ios_base&)) {
    if (OS) *OS << Func;
    return *this;
  }
  */

  Stream operator<<(std::ostream &(*Func)(std::ostream &)) {
    if (OS) {
      Func(*OS);
    }
    return *this;
  }

  template <typename Ty> Stream &operator<<(const Ty &Thing) {
    if (OS) {
      Writer<Ty, sizeof(sfinae_bait(Thing))>::write(*OS, Thing);
    }
    return *this;
  }

private:
  // Implementation details to treat llvm::Value, llvm::Type and their
  // subclasses specially (to pretty-print types).

  static void writeType(std::ostream &OS, const llvm::Type &Ty);
  static void writeValue(std::ostream &OS, const llvm::Value &Ty);

  template <typename Ty, int N> friend struct Writer;
  // error: function template partial specialization is not allowed
  // So I guess type partial specialization + member function will have to do...
  template <typename Ty, int N> struct Writer {
    static void write(std::ostream &OS, const Ty &Thing) { OS << Thing; }
  };

  template <typename Ty> struct Writer<Ty, 1> {
    static void write(std::ostream &OS, const llvm::Type &Thing) {
      Stream::writeType(OS, Thing);
    }
    static void write(std::ostream &OS, const llvm::Value &Thing) {
      Stream::writeValue(OS, Thing);
    }
  };

  // NOT IMPLEMENTED
  char sfinae_bait(const llvm::Type &);
  char sfinae_bait(const llvm::Value &);
  short sfinae_bait(...);
};

extern bool _Logger_enabled;

namespace Logger {

void indent();
void undent();
Stream cout();
void printIndentation();
void println(const char *fmt, ...) IS_PRINTF(1);
void print(const char *fmt, ...) IS_PRINTF(1);
inline void enable() { _Logger_enabled = true; }
inline void disable() { _Logger_enabled = false; }
inline bool enabled() { return _Logger_enabled; }

struct LoggerScope {
  LoggerScope() { Logger::indent(); }
  ~LoggerScope() { Logger::undent(); }
};
}

#define LOG_SCOPE Logger::LoggerScope _logscope;

#define IF_LOG if (Logger::enabled())
