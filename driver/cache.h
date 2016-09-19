//===-- driver/cache.h ------------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_DRIVER_IR2OBJ_CACHE_H
#define LDC_DRIVER_IR2OBJ_CACHE_H

#include <string>

#include "ddmd/arraytypes.h"

namespace llvm {
class Module;
class StringRef;
template <unsigned> class SmallString;
}

namespace cache {

void calculateModuleHash(llvm::Module *m, llvm::SmallString<32> &str);
std::string cacheLookup(llvm::StringRef cacheObjectHash);
std::string cacheObjectFile(llvm::StringRef objectFile,
                            llvm::StringRef cacheObjectHash);
void recoverObjectFile(llvm::StringRef cacheObjectHash,
                       llvm::StringRef objectFile);
void recoverObjectFile(const char *cacheFile, size_t cacheFileLen,
                       const char *objectFile, size_t objectFileLen);
/// Prune the cache to avoid filling up disk space.
///
/// Note: Does nothing for LLVM < 3.7.
void pruneCache();
}

void cacheManifest(const char *hash, const char *cacheObjFile);

#endif
