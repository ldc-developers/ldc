//===-- driver/ir2obj_cache_pruning.h ---------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is adapted from LLVM's lib/Support/CachePruning.h. Therefore,
// this file is distributed under the LLVM license.
// See the LICENSE file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_LLVM_VER >= 307

#ifndef LLVMLDC_SUPPORT_CACHE_PRUNING_H
#define LLVMLDC_SUPPORT_CACHE_PRUNING_H

#include "llvm/ADT/StringRef.h"

namespace llvmldc {

/// Handle pruning a directory provided a path and some options to control what
/// to prune.
class CachePruning {
public:
  /// Prepare to prune \p Path.
  CachePruning(llvm::StringRef Path) : Path(Path) {}

  /// Define the pruning interval. This is intended to be used to avoid scanning
  /// the directory too often. It does not impact the decision of which file to
  /// prune. A value of 0 forces the scan to occurs.
  CachePruning &setPruningInterval(int PruningInterval) {
    Interval = PruningInterval;
    return *this;
  }

  /// Define the expiration for a file. When a file hasn't been accessed for
  /// \p ExpireAfter seconds, it is removed from the cache. A value of 0 disable
  /// the expiration-based pruning.
  CachePruning &setEntryExpiration(unsigned ExpireAfter) {
    Expiration = ExpireAfter;
    return *this;
  }

  /// Define the maximum size for the cache directory, in terms of bytes.
  /// Set to 0 (default) to limit the cache size to the percentage of
  /// the available space on the the disk set by setMaxSize.
  CachePruning &setMaxSizeBytes(uint64_t SizeInBytes) {
    MaxSizeInBytes = SizeInBytes;
    return *this;
  }

  /// Define the maximum size for the cache directory, in terms of percentage of
  /// the available space on the the disk. Set to 100 to indicate no limit, 50
  /// to indicate that the cache size will not be left over half the
  /// available disk space. A value over 100 will be reduced to 100. A value of
  /// 0 disable the size-based pruning.
  /// For LLVM < 3.9 this setting is ignored (the available disk space is not
  /// available).
  CachePruning &setMaxSize(unsigned Percentage) {
    PercentageOfAvailableSpace = std::min(100u, Percentage);
    return *this;
  }

  /// Peform pruning using the supplied options, returns true if pruning
  /// occured, i.e. if PruningInterval was expired.
  bool prune();

private:
  // Options that matches the setters above.
  std::string Path;
  unsigned Expiration = 0;
  unsigned Interval = 0;
  unsigned PercentageOfAvailableSpace = 0;
  uint64_t MaxSizeInBytes = 0;

  bool IsSizeAboveMaximum(uint64_t Size, uint64_t AvailableSpace) {
    bool TooLarge = false;
    if (MaxSizeInBytes > 0)
      TooLarge = Size > MaxSizeInBytes;

#if LDC_LLVM_VER >= 309
    TooLarge = TooLarge ||
               ((100 * Size) / AvailableSpace) > PercentageOfAvailableSpace;
#endif

    return TooLarge;
  }
};

} // namespace llvmldc

#endif

#endif
