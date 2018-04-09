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
struct ParamSlice;

#ifdef _WIN32
#define EXTERNAL __declspec(dllimport) extern
#else
#define EXTERNAL extern
#endif

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
#define JIT_REG_BIND_PAYLOAD                                                   \
  MAKE_JIT_API_CALL(registerBindPayloadImplSo, LDC_DYNAMIC_COMPILE_API_VERSION)
#define JIT_UNREG_BIND_PAYLOAD                                                 \
  MAKE_JIT_API_CALL(unregisterBindPayloadImplSo,                               \
                    LDC_DYNAMIC_COMPILE_API_VERSION)

extern "C" {

// Silence missing-variable-declaration clang warning
extern const void *dynamiccompile_modules_head;

const void *dynamiccompile_modules_head = nullptr;

EXTERNAL void JIT_API_ENTRYPOINT(const void *modlist_head,
                                 const Context *context,
                                 std::size_t contextSize);

EXTERNAL void JIT_REG_BIND_PAYLOAD(void *handle, void *originalFunc,
                                   const ParamSlice *desc, size_t descSize);

EXTERNAL void JIT_UNREG_BIND_PAYLOAD(void *handle);

void rtCompileProcessImpl(const Context *context, std::size_t contextSize) {
  JIT_API_ENTRYPOINT(dynamiccompile_modules_head, context, contextSize);
}

void registerBindPayload(void *handle, void *originalFunc,
                         const ParamSlice *desc, size_t descSize) {
  JIT_REG_BIND_PAYLOAD(handle, originalFunc, desc, descSize);
}

void unregisterBindPayload(void *handle) { JIT_UNREG_BIND_PAYLOAD(handle); }
}
