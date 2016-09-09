//===-- driver/ir2obj_cache_pruning.cpp -------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is adapted from LLVM's lib/Support/CachePruning.cpp. Therefore,
// this file is distributed under the LLVM license.
// See the LICENSE file for details.
//
//===----------------------------------------------------------------------===//
//
// LLVM IR to object code cache directory pruning, to prevent filling up disk
// space.
//
// LLVM's pruning scheme (LLVM 3.9) is modified to support a maximum absolute
// size, and to support LLVM < 3.9 (by disabling the percentage-based size
// limit).
// Important: it also fixes a bug, the LLVM 3.9 code incorrectly deletes files
// without looking at access time when pruning for cache size.
//
//===----------------------------------------------------------------------===//

#if LDC_LLVM_VER >= 307

#include "driver/ir2obj_cache_pruning.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "cache-pruning"

#include <queue>

using namespace llvm;

namespace llvmldc {

/// Write a new timestamp file with the given path. This is used for the pruning
/// interval option.
static void writeTimestampFile(StringRef TimestampFile) {
  std::error_code EC;
  raw_fd_ostream Out(TimestampFile.str(), EC, sys::fs::F_None);
}

/// Prune the cache of files that haven't been accessed in a long time.
bool CachePruning::prune() {
  if (Path.empty())
    return false;

  bool isPathDir;
  if (sys::fs::is_directory(Path, isPathDir))
    return false;

  if (!isPathDir)
    return false;

  if (Expiration == 0 && PercentageOfAvailableSpace == 0) {
    DEBUG(dbgs() << "No pruning settings set, exit early\n");
    // Nothing will be pruned, early exit
    return false;
  }

  // Try to stat() the timestamp file.
  SmallString<128> TimestampFile(Path);
  sys::path::append(TimestampFile, "ldccache.timestamp");
  sys::fs::file_status FileStatus;
  sys::TimeValue CurrentTime = sys::TimeValue::now();
  if (auto EC = sys::fs::status(TimestampFile, FileStatus)) {
    if (EC == errc::no_such_file_or_directory) {
      // If the timestamp file wasn't there, create one now.
      writeTimestampFile(TimestampFile);
    } else {
      // Unknown error?
      return false;
    }
  } else {
    if (Interval) {
      // Check whether the time stamp is older than our pruning interval.
      // If not, do nothing.
      sys::TimeValue TimeStampModTime = FileStatus.getLastModificationTime();
      auto TimeInterval = sys::TimeValue(sys::TimeValue::SecondsType(Interval));
      auto TimeStampAge = CurrentTime - TimeStampModTime;
      if (TimeStampAge <= TimeInterval) {
        DEBUG(dbgs() << "Timestamp file too recent (" << TimeStampAge.seconds()
                     << "s old), do not prune.\n");
        return false;
      }
    }
    // Write a new timestamp file so that nobody else attempts to prune.
    // There is a benign race condition here, if two processes happen to
    // notice at the same time that the timestamp is out-of-date.
    writeTimestampFile(TimestampFile);
  }

  bool ShouldComputeSize =
      (MaxSizeInBytes > 0) || (PercentageOfAvailableSpace > 0);

  // Keep track of space
  struct FileSizeAndAge {
    std::string File;
    uint64_t Size;
    sys::TimeValue Age;

    // Comparison function for priority_queue (return "false" if f1 is ordered
    // first).
    struct OldAgesFirst {
      bool operator()(FileSizeAndAge const &f1, FileSizeAndAge const &f2) {
        return f1.Age < f2.Age;
      }
    };
  };
  std::priority_queue<FileSizeAndAge, std::vector<FileSizeAndAge>,
                      FileSizeAndAge::OldAgesFirst>
      FileSizes;
  uint64_t TotalSize = 0;

  // Helper to add a path to the set of files to consider for size-based
  // pruning, sorted by size.
  auto AddToFileListForSizePruning = [&](StringRef Path, sys::TimeValue Age) {
    if (!ShouldComputeSize)
      return;
    TotalSize += FileStatus.getSize();
    FileSizes.push({Path, FileStatus.getSize(), Age});
  };

  // Walk the entire directory cache, looking for unused files.
  std::error_code EC;
  SmallString<128> CachePathNative;
  sys::path::native(Path, CachePathNative);
  auto TimeExpiration = sys::TimeValue(sys::TimeValue::SecondsType(Expiration));
  // Walk all of the files within this directory.
  for (sys::fs::directory_iterator File(CachePathNative, EC), FileEnd;
       File != FileEnd && !EC; File.increment(EC)) {
    // Do not touch the timestamp.
    if (File->path() == TimestampFile)
      continue;

    // Look at this file. If we can't stat it, there's nothing interesting
    // there.
    if (sys::fs::status(File->path(), FileStatus)) {
      DEBUG(dbgs() << "Ignore " << File->path() << " (can't stat)\n");
      continue;
    }

// If the file hasn't been used recently enough, delete it
#if LDC_LLVM_VER >= 309
    sys::TimeValue FileAccessTime = FileStatus.getLastAccessedTime();
#else
    // getLastAccessedTime is not available in LLVM < 3.9.
    // Resort to last modification time (this is updated everytime LDC does a
    // cache file recovery)
    sys::TimeValue FileAccessTime = FileStatus.getLastModificationTime();
#endif
    auto FileAge = CurrentTime - FileAccessTime;
    if (FileAge > TimeExpiration) {
      DEBUG(dbgs() << "Remove " << File->path() << " (" << FileAge.seconds()
                   << "s old)\n");
      sys::fs::remove(File->path());
      continue;
    }

    // Leave it here for now, but add it to the list of size-based pruning.
    AddToFileListForSizePruning(File->path(), FileAge);
  }

  // Prune for size now if needed
  if (ShouldComputeSize) {
#if LDC_LLVM_VER >= 309
    auto ErrOrSpaceInfo = sys::fs::disk_space(Path);
    if (!ErrOrSpaceInfo) {
      report_fatal_error("Can't get available size");
    }
    sys::fs::space_info SpaceInfo = ErrOrSpaceInfo.get();
    auto AvailableSpace = TotalSize + SpaceInfo.free;
#else
    // The AvailableSpace will be ignored for LLVM < 3.9. This is just to get
    // its type right.
    auto AvailableSpace = TotalSize;
#endif
    DEBUG(dbgs() << "Occupancy: " << ((100 * TotalSize) / AvailableSpace)
                 << "% target is: " << PercentageOfAvailableSpace << "\n");
    // Remove the oldest accessed files first, till we get below the threshold
    while (IsSizeAboveMaximum(TotalSize, AvailableSpace) &&
           !FileSizes.empty()) {
      auto const &FileAndSize = FileSizes.top();
      // Remove the file.
      sys::fs::remove(FileAndSize.File);
      // Update size
      TotalSize -= FileAndSize.Size;
      DEBUG(dbgs() << " - Remove " << FileAndSize.File << " (size "
                   << FileAndSize.Size << "), new occupancy is " << TotalSize
                   << "%\n");
      FileSizes.pop();
    }
  }
  return true;
}

} // namespace llvmldc

#endif
