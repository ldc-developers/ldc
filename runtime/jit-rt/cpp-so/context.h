//===-- context.h - jit support ---------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Jit compilation context, must be in sync with runtimecompile.d.
//
//===----------------------------------------------------------------------===//

#ifndef CONTEXT_H
#define CONTEXT_H

#include <cstddef> //size_t

enum class DumpStage : int {
  OriginalModule = 0,
  MergedModule = 1,
  OptimizedModule = 2,
  FinalAsm = 3
};

enum {
    ApiVersion = LDC_DYNAMIC_COMPILE_API_VERSION
};

#define MAKE_JIT_API_CALL_IMPL(prefix, version) prefix##version
#define MAKE_JIT_API_CALL(prefix, version) \
  MAKE_JIT_API_CALL_IMPL(prefix, version)
#define JIT_API_ENTRYPOINT MAKE_JIT_API_CALL(rtCompileProcessImplSo, \
  LDC_DYNAMIC_COMPILE_API_VERSION)

typedef void (*InterruptPointHandlerT)(void *, const char *action,
                                       const char *object);
typedef void (*FatalHandlerT)(void *, const char *reason);
typedef void (*DumpHandlerT)(void *, DumpStage stage, const char *str,
                             std::size_t len);

struct Context final {
  unsigned optLevel = 0;
  unsigned sizeLevel = 0;
  InterruptPointHandlerT interruptPointHandler = nullptr;
  void *interruptPointHandlerData = nullptr;
  FatalHandlerT fatalHandler = nullptr;
  void *fatalHandlerData = nullptr;
  DumpHandlerT dumpHandler = nullptr;
  void *dumpHandlerData = nullptr;
  bool genInstrumentation = false;
  bool useInstrumentation = false;
};

#endif // CONTEXT_H
