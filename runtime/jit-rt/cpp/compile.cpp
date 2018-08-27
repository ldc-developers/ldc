//===-- compile.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Jit runtime - executable part.
// Defines jit modules list head and and access jit shared library entry point.
//
//===----------------------------------------------------------------------===//

#include <cstddef> // size_t

struct Context;

#ifdef _WIN32
#define EXTERNAL __declspec(dllimport) extern
#else
#define EXTERNAL extern
#endif

#define MAKE_JIT_API_CALL_IMPL(prefix, version) prefix##version
#define MAKE_JIT_API_CALL(prefix, version)                                     \
  MAKE_JIT_API_CALL_IMPL(prefix, version)
#define JIT_API_ENTRYPOINT                                                     \
  MAKE_JIT_API_CALL(rtCompileProcessImplSo, LDC_DYNAMIC_COMPILE_API_VERSION)

extern "C" {

// Silence missing-variable-declaration clang warning
extern const void *dynamiccompile_modules_head;

const void *dynamiccompile_modules_head = nullptr;

EXTERNAL void JIT_API_ENTRYPOINT(const void *modlist_head,
                                 const Context *context,
                                 std::size_t contextSize);

void rtCompileProcessImpl(const Context *context, std::size_t contextSize) {
  JIT_API_ENTRYPOINT(dynamiccompile_modules_head, context, contextSize);
}
}
