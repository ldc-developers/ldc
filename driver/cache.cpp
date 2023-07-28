//===-- driver/cache.cpp --------------------------------------------------===//
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

#include "driver/cache.h"

#include "dmd/errors.h"
#include "dmd/target.h"
#include "driver/cache_pruning.h"
#include "driver/cl_options.h"
#include "driver/cl_options_sanitizers.h"
#include "driver/ldc-version.h"
#include "gen/logger.h"
#include "gen/optimizer.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/Chrono.h"
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

#if LDC_POSIX
#include <unistd.h>
#include <errno.h>

static std::error_code createHardLink(const char *to, const char *from) {
  if (link(to, from) == 0)
    return std::error_code(0, std::system_category());
  else
    return std::error_code(errno, std::system_category());
}

static std::error_code createSymLink(const char *to, const char *from) {
  if (symlink(to, from) == 0)
    return std::error_code(0, std::system_category());
  else
    return std::error_code(errno, std::system_category());
}
#elif _WIN32
#include <windows.h>
namespace llvm {
namespace sys {
namespace windows {
// Fwd declaration to an internal LLVM function.
std::error_code widenPath(const llvm::Twine &Path8,
                          llvm::SmallVectorImpl<wchar_t> &Path16,
                          size_t MaxPathLen = MAX_PATH);
}
} // namespace sys
} // namespace llvm

namespace {
template <typename FType>
std::error_code createLink(FType f, const char *to, const char *from) {
  //===----------------------------------------------------------------------===//
  //
  // Code copied from LLVM 3.9 llvm/Support/Windows/Path.inc, distributed under
  // the University of Illinois Open Source License. See LICENSE for details.
  //
  //===----------------------------------------------------------------------===//

  using llvm::sys::windows::widenPath;

  llvm::SmallVector<wchar_t, 128> wide_from;
  llvm::SmallVector<wchar_t, 128> wide_to;
  if (auto errorcode = widenPath(from, wide_from))
    return errorcode;
  if (auto errorcode = widenPath(to, wide_to))
    return errorcode;

  if (!(*f)(wide_from.begin(), wide_to.begin(), NULL))
    return std::error_code(GetLastError(), std::system_category());;

  return std::error_code(0, std::system_category());
}
}

static std::error_code createHardLink(const char *to, const char *from) {
  return createLink(&CreateHardLinkW, to, from);
}

static std::error_code createSymLink(const char *to, const char *from) {
  return createLink(&CreateSymbolicLinkW, to, from);
}
#endif

namespace {

// Options for the cache pruning algorithm
llvm::cl::opt<bool> pruneEnabled("cache-prune",
                                 llvm::cl::desc("Enable cache pruning."));
llvm::cl::opt<unsigned long long> pruneSizeLimitInBytes(
    "cache-prune-maxbytes",
    llvm::cl::desc("Sets the maximum cache size to <size> bytes. Implies "
                   "-cache-prune."),
    llvm::cl::value_desc("size"), llvm::cl::init(0));
llvm::cl::opt<unsigned> pruneInterval(
    "cache-prune-interval",
    llvm::cl::desc("Sets the cache pruning interval to <dur> seconds "
                   "(default: 20 min). Set to 0 to force pruning. Implies "
                   "-cache-prune."),
    llvm::cl::value_desc("dur"), llvm::cl::init(20 * 60));
llvm::cl::opt<unsigned> pruneExpiration(
    "cache-prune-expiration",
    llvm::cl::desc("Sets the pruning expiration time of cache files to "
                   "<dur> seconds (default: 1 week). Implies -cache-prune."),
    llvm::cl::value_desc("dur"), llvm::cl::init(7 * 24 * 3600));
llvm::cl::opt<unsigned> pruneSizeLimitPercentage(
    "cache-prune-maxpercentage",
    llvm::cl::desc(
        "Sets the cache size limit to <perc> percent of the available "
        "space (default: 75%). Implies -cache-prune."),
    llvm::cl::value_desc("perc"), llvm::cl::init(75));

enum class RetrievalMode { Copy, HardLink, AnyLink, SymLink };
llvm::cl::opt<RetrievalMode> cacheRecoveryMode(
    "cache-retrieval", llvm::cl::ZeroOrMore,
    llvm::cl::desc("Set the cache retrieval mechanism (default: copy)."),
    llvm::cl::init(RetrievalMode::Copy),
    llvm::cl::values(
        clEnumValN(RetrievalMode::Copy, "copy",
                   "Make a copy of the cache file"),
        clEnumValN(RetrievalMode::HardLink, "hardlink",
                   "Create a hard link to the cache file (recommended)"),
        clEnumValN(
            RetrievalMode::AnyLink, "link",
            "Equal to 'hardlink' on Windows, but 'symlink' on Unix and OS X"),
        clEnumValN(RetrievalMode::SymLink, "symlink",
                   "Create a symbolic link to the cache file")));

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

llvm::sys::TimePoint<std::chrono::seconds> getTimeNow() {
  using namespace std::chrono;
  return time_point_cast<seconds>(system_clock::now());
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
  filePath = opts::cacheDir;
  llvm::sys::path::append(
      filePath, llvm::Twine("ircache_") + cacheObjectHash + "." +
                    llvm::StringRef(target.obj_ext.ptr, target.obj_ext.length));
}

// Output to `hash_os` all commandline flags, and try to skip the ones that have
// no influence on the object code output. The cmdline flags need to be added
// to the ir2obj cache hash to uniquely identify the object file output.
// Because the compiler version is part of the hash, differences in the
// default settings between compiler versions are already taken care of.
// (Note: config and response files may also add compiler flags.)
void outputIR2ObjRelevantCmdlineArgs(llvm::raw_ostream &hash_os) {
  // Use a "whitelist" of cmdline args that do not need to be added to the hash,
  // and add all others. There is no harm (other than missed cache
  // opportunities) in adding commandline arguments that also change the hashed
  // IR, which simplifies the code here.
  // The code does not deal well with options specified without equals sign, and
  // will add those to the hash, resulting in missed cache opportunities.

  auto it = opts::allArguments.begin();
  auto end_it = opts::allArguments.end();
  // The first argument is the compiler executable filename: we can skip it.
  ++it;
  for (; it != end_it; ++it) {
    const char *arg = *it;
    if (!arg || !arg[0])
      continue;

    // Out of pre-caution, all arguments that are not prefixed with '-' are
    // added to the hash. Such an argument could be a source file "foo.d", but
    // also a value for the previous argument when the equals sign is omitted,
    // for example: "-code-model default" becomes "-code-model" "default".
    // It results in missed cache opportunities. :(
    if (arg[0] == '-') {
      if (arg[1] == 'O') {
        // We deal with -O later ("-O" and "-O3" should hash equally, "" and
        // "-O0" too)
        continue;
      }
      if (arg[1] == 'c' && !arg[2])
        continue;
      // All options starting with these characters can be ignored (LLVM does
      // not have options starting with capitals)
      if (arg[1] == 'D' || arg[1] == 'H' || arg[1] == 'I' || arg[1] == 'J' ||
          arg[1] == 'L' || arg[1] == 'X')
        continue;
      if (arg[1] == 'd' || arg[1] == 'v' || arg[1] == 'w') {
        // LLVM options are long, so short options starting with 'v' or 'w' can
        // be ignored.
        unsigned len = 2;
        for (; len < 11; ++len)
          if (!arg[len])
            break;
        if (len < 11)
          continue;
      }
      // "-of..." can be ignored
      if (arg[1] == 'o' && arg[2] == 'f')
        continue;
      // "-od..." can be ignored
      if (arg[1] == 'o' && arg[2] == 'd')
        continue;
      // All  "-cache..." options can be ignored
      if (strncmp(arg + 1, "cache", 5) == 0)
        continue;
      // Ignore "-lib"
      if (arg[1] == 'l' && arg[2] == 'i' && arg[3] == 'b' && !arg[4])
        continue;
      // All effects of -d-version... are already included in the IR hash.
      if (strncmp(arg + 1, "d-version", 9) == 0)
        continue;
      // All effects of -unittest are already included in the IR hash.
      if (strcmp(arg + 1, "unittest") == 0) {
        continue;
      }

      // All arguments following -run can safely be ignored
      if (strcmp(arg + 1, "run") == 0) {
        break;
      }
    }

    // If we reach here, add the argument to the hash.
    hash_os << arg;
  }

  // Adding these options to the hash should not be needed after adding all
  // cmdline args. We keep this code here however, in case we find a different
  // solution for dealing with LLVM commandline flags. See GH #1773.
  // Also, having these options explicitly added to the hash protects against
  // the possibility of different default settings on different platforms (while
  // sharing the cache).
  outputOptimizationSettings(hash_os);
  opts::outputSanitizerSettings(hash_os);
  hash_os << opts::getCPUStr();
  hash_os << opts::getFeaturesStr();
  hash_os << opts::floatABI;
  const auto relocModel = opts::getRelocModel();
#if LDC_LLVM_VER >= 1600
  if (relocModel.has_value())
    hash_os << relocModel.value();
#else
  if (relocModel.hasValue())
    hash_os << relocModel.getValue();
#endif
  const auto codeModel = opts::getCodeModel();
#if LDC_LLVM_VER >= 1600
  if (codeModel.has_value())
    hash_os << codeModel.value();
#else
  if (codeModel.hasValue())
    hash_os << codeModel.getValue();
#endif

  const auto framePointerUsage = opts::framePointerUsage();
#if LDC_LLVM_VER >= 1600
  if (framePointerUsage.has_value())
    hash_os << static_cast<int>(framePointerUsage.value());
#else
  if (framePointerUsage.hasValue())
    hash_os << static_cast<int>(framePointerUsage.getValue());
#endif
}

// Output to `hash_os` all environment flags that influence object code output
// in ways that are not observable in the pre-LLVM passes IR used for hashing.
void outputIR2ObjRelevantEnvironmentOpts(llvm::raw_ostream &hash_os) {
  // There are no relevant environment options at the moment.
}

} // anonymous namespace

namespace cache {

void calculateModuleHash(llvm::Module *m, llvm::SmallString<32> &str) {
  raw_hash_ostream hash_os;

  // Let hash depend on the compiler version:
  hash_os << ldc::ldc_version << ldc::dmd_version << ldc::llvm_version
          << ldc::built_with_Dcompiler_version;

  // Let hash depend on compile flags that change the outputted obj file,
  // but whose changes are not always observable in the pre-optimized IR used
  // for hashing:
  outputIR2ObjRelevantCmdlineArgs(hash_os);
  outputIR2ObjRelevantEnvironmentOpts(hash_os);

  llvm::WriteBitcodeToFile(*m, hash_os);
  hash_os.resultAsString(str);
  IF_LOG Logger::println("Module's LLVM bitcode hash is: %s", str.c_str());
}

std::string cacheLookup(llvm::StringRef cacheObjectHash) {
  if (opts::cacheDir.empty())
    return "";

  if (!llvm::sys::fs::exists(opts::cacheDir)) {
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
  if (opts::cacheDir.empty())
    return;

  if (!llvm::sys::fs::exists(opts::cacheDir)) {
    if (auto errorcode = llvm::sys::fs::create_directories(opts::cacheDir)) {
      error(Loc(), "Unable to create cache directory: %s (errno %d: %s)",
            opts::cacheDir.c_str(), errorcode.value(),
            errorcode.message().c_str());
      fatal();
    }
  }

  // To prevent bad cache files, add files to the cache atomically: first copy
  // to a temporary file and then rename that temp file to the cache entry
  // filename (rename is atomic).

  llvm::SmallString<128> cacheFile;
  storeCacheFileName(cacheObjectHash, cacheFile);

  llvm::SmallString<128> tempFile;
  if (auto errorcode = llvm::sys::fs::createUniqueFile(
          llvm::Twine(cacheFile) + ".tmp%%%%%%%", tempFile)) {
    error(
        Loc(),
        "Could not create name of temporary file in the cache (errno %d: %s)",
        errorcode.value(), errorcode.message().c_str());
    fatal();
  }

  IF_LOG Logger::println("Copy object file to temp file: %s to %s",
                         objectFile.str().c_str(), tempFile.c_str());
  if (auto errorcode = llvm::sys::fs::copy_file(objectFile, tempFile.c_str())) {
    error(Loc(),
          "Failed to copy object file to cache: %s to %s (errno %d: %s)",
          objectFile.str().c_str(), tempFile.c_str(), errorcode.value(),
          errorcode.message().c_str());
    fatal();
  }
  IF_LOG Logger::println("Rename temp file to cache file: %s to %s",
                         tempFile.c_str(), cacheFile.c_str());
  if (auto errorcode =
          llvm::sys::fs::rename(tempFile.c_str(), cacheFile.c_str())) {
    error(Loc(),
          "Failed to rename temp file to cache file: %s to %s (errno %d: %s)",
          tempFile.c_str(), cacheFile.c_str(), errorcode.value(),
          errorcode.message().c_str());
    fatal();
  }
}

void recoverObjectFile(llvm::StringRef cacheObjectHash,
                       llvm::StringRef objectFile) {
  llvm::SmallString<128> cacheFile;
  storeCacheFileName(cacheObjectHash, cacheFile);

  // Remove the potentially pre-existing output file.
  llvm::sys::fs::remove(objectFile);

  switch (cacheRecoveryMode) {
  case RetrievalMode::Copy: {
    IF_LOG Logger::println("Copy cached object file: %s -> %s",
                           cacheFile.c_str(), objectFile.str().c_str());
    if (auto errorcode =
            llvm::sys::fs::copy_file(cacheFile.c_str(), objectFile)) {
      error(Loc(), "Failed to copy the cached file: %s -> %s (errno %d: %s)",
            cacheFile.c_str(), objectFile.str().c_str(), errorcode.value(),
            errorcode.message().c_str());
      fatal();
    }
  } break;
  case RetrievalMode::HardLink: {
    IF_LOG Logger::println("HardLink output to cached object file: %s -> %s",
                           objectFile.str().c_str(), cacheFile.c_str());
    if (auto errorcode =
            createHardLink(cacheFile.c_str(), objectFile.str().c_str())) {
      error(Loc(),
            "Failed to create a hard link to the cached file: %s -> %s (errno "
            "%d: %s)",
            cacheFile.c_str(), objectFile.str().c_str(), errorcode.value(),
            errorcode.message().c_str());
      fatal();
    }
  } break;
  case RetrievalMode::AnyLink: {
    IF_LOG Logger::println("Link output to cached object file: %s -> %s",
                           objectFile.str().c_str(), cacheFile.c_str());
    if (auto errorcode =
            llvm::sys::fs::create_link(cacheFile.c_str(), objectFile)) {
      error(
          Loc(),
          "Failed to create a link to the cached file: %s -> %s (errno %d: %s)",
          cacheFile.c_str(), objectFile.str().c_str(), errorcode.value(),
          errorcode.message().c_str());
      fatal();
    }
  } break;
  case RetrievalMode::SymLink: {
    IF_LOG Logger::println("SymLink output to cached object file: %s -> %s",
                           objectFile.str().c_str(), cacheFile.c_str());
    if (auto errorcode =
            createSymLink(cacheFile.c_str(), objectFile.str().c_str())) {
      error(Loc(),
            "Failed to create a symbolic link to the cached file: %s -> %s "
            "(errno %d: %s)",
            cacheFile.c_str(), objectFile.str().c_str(), errorcode.value(),
            errorcode.message().c_str());
      fatal();
    }
  } break;
  }

  // We reset the modification time to "now" such that the pruning algorithm
  // sees that the file should be kept over older files.
  // On some systems the last accessed time is not automatically updated so set
  // it explicitly here. Because the file will really only be accessed later
  // during linking, it's not perfect but it's the best we can do.
  {
    int FD;
    if (llvm::sys::fs::openFileForWrite(cacheFile.c_str(), FD,
                                        llvm::sys::fs::CD_OpenExisting,
                                        llvm::sys::fs::OF_Append)) {
      error(Loc(), "Failed to open the cached file for writing: %s",
            cacheFile.c_str());
      fatal();
    }

    if (llvm::sys::fs::setLastAccessAndModificationTime(FD, getTimeNow())) {
      error(Loc(), "Failed to set the cached file modification time: %s",
            cacheFile.c_str());
      fatal();
    }

    close(FD);
  }
}

void pruneCache() {
  if (!opts::cacheDir.empty() && isPruningEnabled()) {
    ::pruneCache(opts::cacheDir.data(), opts::cacheDir.size(), pruneInterval,
                 pruneExpiration, pruneSizeLimitInBytes,
                 pruneSizeLimitPercentage);
  }
}
} // namespace cache
