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
#define EXTERNAL __attribute__((visibility("default"))) extern
#endif

template <typename T> struct Slice final {
  size_t len;
  T *data;
};

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

struct DynamicCompilerContext;

extern "C" {

// Silence missing-variable-declaration clang warning
extern const void *dynamiccompile_modules_head;

const void *dynamiccompile_modules_head = nullptr;

EXTERNAL void JIT_API_ENTRYPOINT(const void *modlist_head,
                                 const Context *context,
                                 std::size_t contextSize);

EXTERNAL void JIT_REG_BIND_PAYLOAD(DynamicCompilerContext *context,
                                   void *handle, void *originalFunc,
                                   const ParamSlice *desc, size_t descSize);

EXTERNAL void JIT_UNREG_BIND_PAYLOAD(DynamicCompilerContext *context,
                                     void *handle);

EXTERNAL DynamicCompilerContext *JIT_CREATE_COMPILER_CONTEXT();

EXTERNAL void JIT_DESTROY_COMPILER_CONTEXT(DynamicCompilerContext *context);

EXTERNAL bool JIT_SET_OPTS(const Slice<Slice<const char>> *args,
                           void (*errs)(void *, const char *, size_t),
                           void *errsContext);

void rtCompileProcessImpl(const Context *context, std::size_t contextSize) {
  JIT_API_ENTRYPOINT(dynamiccompile_modules_head, context, contextSize);
}

void registerBindPayload(DynamicCompilerContext *context, void *handle,
                         void *originalFunc, const ParamSlice *desc,
                         size_t descSize) {
  JIT_REG_BIND_PAYLOAD(context, handle, originalFunc, desc, descSize);
}

void unregisterBindPayload(DynamicCompilerContext *context, void *handle) {
  JIT_UNREG_BIND_PAYLOAD(context, handle);
}

DynamicCompilerContext *createDynamicCompilerContextImpl() {
  return JIT_CREATE_COMPILER_CONTEXT();
}
void destroyDynamicCompilerContextImpl(DynamicCompilerContext *context) {
  JIT_DESTROY_COMPILER_CONTEXT(context);
}

bool setDynamicCompilerOpts(const Slice<Slice<const char>> *args,
                            void (*errs)(void *, const char *, size_t),
                            void *errsContext) {
  return JIT_SET_OPTS(args, errs, errsContext);
}
}
