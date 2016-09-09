//===-- driver/ir2obj_cache.cpp -------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Contains LLVM IR to object code cache functionality.
//
// After LLVM IR codegen, the LLVM IR module is hashed for lookup in the cache
// directory. If the cache directory contains the object file <hash>.o,
// that file is used and machine code gen is skipped entirely. If the cache
// doesn't contain that file, machine codegen happens as normal and the object
// code is added to the cache.
// The goal is to speed up successive builds of large codebases after minor
// changes that trigger recompilation of many files but with little effective
// changes (in the extreme case, adding a comment in a "globals.d").
//
// The current limitation is that hashing and cache look-up are done with
// whole-module granularity. Future work could attempt to do this for module
// "fragments".
//
// The hash depends on the IR code (obviously), but also on the compiler+LLVM
// versions and several compile flags (e.g. -O*, -mcpu, and -mattr).
//
//===----------------------------------------------------------------------===//

#include "driver/ir2obj_cache.h"

#include "ddmd/errors.h"
#include "driver/cl_options.h"
#include "driver/ldc-version.h"
#include "driver/ir2obj_cache_pruning.h"
#include "gen/logger.h"
#include "gen/optimizer.h"

#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

// Include close() declaration.
#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

namespace {

// Options for the cache pruning algorithm
llvm::cl::opt<bool> pruneEnabled("ir2obj-cache-prune",
                                 llvm::cl::desc("Enable cache pruning."),
                                 llvm::cl::ZeroOrMore);
llvm::cl::opt<unsigned long long> pruneSizeLimitInBytes(
    "ir2obj-cache-prune-maxbytes",
    llvm::cl::desc("Sets the maximum cache size to <size> bytes. Implies "
                   "-ir2obj-cache-prune."),
    llvm::cl::value_desc("size"), llvm::cl::init(0));
llvm::cl::opt<unsigned> pruneInterval(
    "ir2obj-cache-prune-interval",
    llvm::cl::desc("Sets the cache pruning interval to <dur> seconds "
                   "(default: 20 min). Set to 0 to force pruning. Implies "
                   "-ir2obj-cache-prune."),
    llvm::cl::value_desc("dur"), llvm::cl::init(20 * 60));
llvm::cl::opt<unsigned> pruneExpiration(
    "ir2obj-cache-prune-expiration",
    llvm::cl::desc(
        "Sets the pruning expiration time of cache files to "
        "<dur> seconds (default: 1 week). Implies -ir2obj-cache-prune."),
    llvm::cl::value_desc("dur"), llvm::cl::init(7 * 24 * 3600));
llvm::cl::opt<unsigned> pruneSizeLimitPercentage(
    "ir2obj-cache-prune-maxpercentage",
    llvm::cl::desc(
        "Sets the cache size limit to <perc> percent of the available "
        "space (default: 75%). Implies -ir2obj-cache-prune."),
    llvm::cl::value_desc("perc"), llvm::cl::init(75));

bool isPruningEnabled() {
  if (pruneEnabled)
    return true;

  // Specifying cache pruning parameters implies enabling pruning.
  if ((pruneSizeLimitInBytes.getNumOccurrences() > 0) ||
      (pruneInterval.getNumOccurrences() > 0) ||
      (pruneExpiration.getNumOccurrences() > 0) ||
      (pruneSizeLimitPercentage.getNumOccurrences() > 0))
    return true;

  return false;
}

/// A raw_ostream that creates a hash of what is written to it.
/// This class does not encounter output errors.
/// There is no buffering and the hasher can be used at any time.
class raw_hash_ostream : public llvm::raw_ostream {
  llvm::MD5 hasher;

  /// See raw_ostream::write_impl.
  void write_impl(const char *ptr, size_t size) override {
    hasher.update(
        llvm::ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(ptr), size));
  }

  uint64_t current_pos() const override { return 0; }

public:
  raw_hash_ostream() { SetUnbuffered(); }
  ~raw_hash_ostream() override {}

  void flush() = delete;

  void finalResult(llvm::MD5::MD5Result &result) { hasher.final(result); }
  void resultAsString(llvm::SmallString<32> &str) {
    llvm::MD5::MD5Result result;
    hasher.final(result);
    llvm::MD5::stringifyResult(result, str);
  }
};

void storeCacheFileName(llvm::StringRef cacheObjectHash,
                        llvm::SmallString<128> &filePath) {
  filePath = opts::ir2objCacheDir;
  llvm::sys::path::append(filePath, llvm::Twine("ircache_") + cacheObjectHash +
                                        "." + global.obj_ext);
}
}

namespace ir2obj {

void calculateModuleHash(llvm::Module *m, llvm::SmallString<32> &str) {
  raw_hash_ostream hash_os;

  // Let hash depend on the compiler version:
  hash_os << global.ldc_version << global.version << global.llvm_version
          << ldc::built_with_Dcompiler_version;

  // Let hash depend on a few compile flags that change the outputted obj file,
  // but whose changes are not always observable in the IR:
  hash_os << codeGenOptLevel();
  hash_os << opts::mCPU;
  for (auto &attr : opts::mAttrs) {
    hash_os << attr;
  }
  hash_os << opts::mFloatABI;
  hash_os << opts::mRelocModel;
  hash_os << opts::mCodeModel;
  hash_os << opts::disableFpElim;

  llvm::WriteBitcodeToFile(m, hash_os);
  hash_os.resultAsString(str);
  IF_LOG Logger::println("Module's LLVM bitcode hash is: %s", str.c_str());
}

std::string cacheLookup(llvm::StringRef cacheObjectHash) {
  if (opts::ir2objCacheDir.empty())
    return "";

  if (!llvm::sys::fs::exists(opts::ir2objCacheDir)) {
    IF_LOG Logger::println("Cache directory does not exist, no object found.");
    return "";
  }

  llvm::SmallString<128> filePath;
  storeCacheFileName(cacheObjectHash, filePath);
  if (llvm::sys::fs::exists(filePath.c_str())) {
    IF_LOG Logger::println("Cache object found! %s", filePath.c_str());
    return filePath.str().str();
  }

  IF_LOG Logger::println("Cache object not found.");
  return "";
}

void cacheObjectFile(llvm::StringRef objectFile,
                     llvm::StringRef cacheObjectHash) {
  if (opts::ir2objCacheDir.empty())
    return;

  if (!llvm::sys::fs::exists(opts::ir2objCacheDir) &&
      llvm::sys::fs::create_directory(opts::ir2objCacheDir)) {
    error(Loc(), "Unable to create cache directory: %s",
          opts::ir2objCacheDir.c_str());
    fatal();
  }

  llvm::SmallString<128> cacheFile;
  storeCacheFileName(cacheObjectHash, cacheFile);

  IF_LOG Logger::println("Copy object file to cache: %s to %s",
                         objectFile.str().c_str(), cacheFile.c_str());
  if (llvm::sys::fs::copy_file(objectFile, cacheFile.c_str())) {
    error(Loc(), "Failed to copy object file to cache: %s to %s",
          objectFile.str().c_str(), cacheFile.c_str());
    fatal();
  }
}

void recoverObjectFile(llvm::StringRef cacheObjectHash,
                       llvm::StringRef objectFile) {
  llvm::SmallString<128> cacheFile;
  storeCacheFileName(cacheObjectHash, cacheFile);

  // Remove the potentially pre-existing output file.
  llvm::sys::fs::remove(objectFile);

  IF_LOG Logger::println("SymLink output to cached object file: %s -> %s",
                         objectFile.str().c_str(), cacheFile.c_str());
  if (llvm::sys::fs::create_link(cacheFile.c_str(), objectFile)) {
    error(Loc(), "Failed to create a symlink to the cached file: %s -> %s",
          cacheFile.c_str(), objectFile.str().c_str());
    fatal();
  }

  // We reset the modification time to "now" such that the pruning algorithm
  // sees that the file should be kept over older files.
  // On some systems the last accessed time is not automatically updated so set
  // it explicitly here. Because the file will really only be accessed later
  // during linking, it's not perfect but it's the best we can do.
  {
    int FD;
    if (llvm::sys::fs::openFileForWrite(cacheFile.c_str(), FD,
                                        llvm::sys::fs::F_Append)) {
      error(Loc(), "Failed to open the cached file for writing: %s",
            cacheFile.c_str());
      fatal();
    }

    if (llvm::sys::fs::setLastModificationAndAccessTime(
            FD, llvm::sys::TimeValue::now())) {
      error(Loc(), "Failed to set the cached file modification time: %s",
            cacheFile.c_str());
      fatal();
    }

    close(FD);
  }
}

void pruneCache() {
  if (!opts::ir2objCacheDir.empty() && isPruningEnabled()) {
    ::pruneCache(opts::ir2objCacheDir.data(), opts::ir2objCacheDir.size(),
                 pruneInterval, pruneExpiration, pruneSizeLimitInBytes,
                 pruneSizeLimitPercentage);
  }
}
}