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

extern "C" {

// Silence missing-variable-declaration clang warning
extern const void *dynamiccompile_modules_head;

const void *dynamiccompile_modules_head = nullptr;
#ifdef _WIN32
__declspec(dllimport)
#endif
    extern void rtCompileProcessImplSo(const void *modlist_head,
                                       const Context *context,
                                       std::size_t contextSize);

void rtCompileProcessImpl(const Context *context, std::size_t contextSize) {
  rtCompileProcessImplSo(dynamiccompile_modules_head, context, contextSize);
}
}
