//===-- driver/cache_pruning.h ----------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_DRIVER_IR2OBJ_CACHE_PRUNING_H
#define LDC_DRIVER_IR2OBJ_CACHE_PRUNING_H

#if __LP64__
using d_ulong = unsigned long;
#else
using d_ulong = unsigned long long;
#endif

void pruneCache(const char *cacheDirectoryPtr, size_t cacheDirectoryLen,
                uint32_t pruneIntervalSeconds, uint32_t expireIntervalSeconds,
                d_ulong sizeLimitBytes, uint32_t sizeLimitPercentage);

#endif
