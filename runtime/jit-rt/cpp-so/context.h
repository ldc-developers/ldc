//===-- context.h - jit support ---------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Jit compilation context, must be in sync with dynamiccompile.d.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef> //size_t
#include <cstdint>

#include "param_slice.h"

#include "slice.h"

enum class DumpStage : int {
  OriginalModule = 0,
  MergedModule = 1,
  OptimizedModule = 2,
  FinalAsm = 3
};

enum { ApiVersion = LDC_DYNAMIC_COMPILE_API_VERSION };

#ifdef _WIN32
#define EXTERNAL __declspec(dllexport)
#else
#define EXTERNAL __attribute__((visibility("default")))
#endif

#define MAKE_JIT_API_CALL_IMPL(prefix, version) prefix##version
#define MAKE_JIT_API_CALL_IMPL1(prefix, version)                               \
  MAKE_JIT_API_CALL_IMPL(prefix, version)
#define MAKE_JIT_API_CALL(call)                                                \
  MAKE_JIT_API_CALL_IMPL1(call, LDC_DYNAMIC_COMPILE_API_VERSION)

#define JIT_API_ENTRYPOINT MAKE_JIT_API_CALL(rtCompileProcessImplSo)
#define JIT_REG_BIND_PAYLOAD MAKE_JIT_API_CALL(registerBindPayloadImplSo)
#define JIT_UNREG_BIND_PAYLOAD MAKE_JIT_API_CALL(unregisterBindPayloadImplSo)
#define JIT_CREATE_COMPILER_CONTEXT                                            \
  MAKE_JIT_API_CALL(createDynamicCompilerContextSo)
#define JIT_DESTROY_COMPILER_CONTEXT                                           \
  MAKE_JIT_API_CALL(destroyDynamicCompilerContextSo)
#define JIT_SET_OPTS MAKE_JIT_API_CALL(setDynamicCompilerOptsImpl)

typedef void (*InterruptPointHandlerT)(void *, const char *action,
                                       const char *object);
typedef void (*FatalHandlerT)(void *, const char *reason);
typedef void (*DumpHandlerT)(void *, DumpStage stage, const char *str,
                             std::size_t len);

class DynamicCompilerContext;

struct Context final {
  unsigned optLevel = 0;
  unsigned sizeLevel = 0;
  InterruptPointHandlerT interruptPointHandler = nullptr;
  void *interruptPointHandlerData = nullptr;
  FatalHandlerT fatalHandler = nullptr;
  void *fatalHandlerData = nullptr;
  DumpHandlerT dumpHandler = nullptr;
  void *dumpHandlerData = nullptr;
  DynamicCompilerContext *compilerContext = nullptr;
};
