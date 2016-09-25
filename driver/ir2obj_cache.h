//===-- driver/ir2obj_cache.h -----------------------------------*- C++ -*-===//
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

namespace llvm {
class Module;
class StringRef;
template <unsigned> class SmallString;
}

namespace ir2obj {

void calculateModuleHash(llvm::Module *m, llvm::SmallString<32> &str);
std::string cacheLookup(llvm::StringRef cacheObjectHash);
void cacheObjectFile(llvm::StringRef objectFile,
                     llvm::StringRef cacheObjectHash);
void recoverObjectFile(llvm::StringRef cacheObjectHash,
                       llvm::StringRef objectFile);

/// Prune the cache to avoid filling up disk space.
///
/// Note: Does nothing for LLVM < 3.7.
void pruneCache();
}

#endif
