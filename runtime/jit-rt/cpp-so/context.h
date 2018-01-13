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

typedef void (*InterruptPointHandlerT)(void *, const char *action,
                                       const char *object);
typedef void (*FatalHandlerT)(void *, const char *reason);
typedef void (*DumpHandlerT)(void *, DumpStage stage, const char *str,
                             std::size_t len);
struct Slice
{
    const void* data;
    std::size_t len;
};
typedef void (*LoadCacheSinkT)(void *, const Slice &slice);
typedef void (*LoadCacheHandlerT)(void *, const char *desc, void *,
                                  LoadCacheSinkT sink);
typedef void (*SaveCacheHandlerT)(void *, const char *desc, const Slice &slice);

struct Context final {
  unsigned optLevel = 0;
  unsigned sizeLevel = 0;
  InterruptPointHandlerT interruptPointHandler = nullptr;
  void *interruptPointHandlerData = nullptr;
  FatalHandlerT fatalHandler = nullptr;
  void *fatalHandlerData = nullptr;
  DumpHandlerT dumpHandler = nullptr;
  void *dumpHandlerData = nullptr;
  LoadCacheHandlerT loadCacheHandler = nullptr;
  void *loadCacheHandlerData = nullptr;
  SaveCacheHandlerT saveCacheHandler = nullptr;
  void *saveCacheHandlerData = nullptr;
};

#endif // CONTEXT_H
