//===-- driver/cache_pruning.d ------------------------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Implements cache pruning scheme.
// 0. Check that the cache exists.
// 1. Check that minimum pruning interval has passed.
// 2. Prune files that have passed the expiry duration.
// 3. Prune files to reduce total cache size to below a set limit.
//
// This file is imported by the ldc-prune-cache tool and should therefore depend
// on as little LDC code as possible (currently none).
//
//===----------------------------------------------------------------------===//

module driver.cache_pruning;

import std.file;
import std.datetime: Clock, dur, Duration, SysTime;

// Creates a CachePruner and performs the pruning.
// This function is meant to take care of all C++ interfacing.
extern (C++) void pruneCache(const(char)* cacheDirectoryPtr,
    size_t cacheDirectoryLen, uint pruneIntervalSeconds,
    uint expireIntervalSeconds, ulong sizeLimitBytes, uint sizeLimitPercentage)
{
    import std.conv: to;

    auto pruner = CachePruner(to!(string)(cacheDirectoryPtr[0 .. cacheDirectoryLen]),
        pruneIntervalSeconds, expireIntervalSeconds, sizeLimitBytes, sizeLimitPercentage);

    pruner.doPrune();
}

void writeEmptyFile(string filename)
{
    import std.stdio: File;
    auto f = File(filename, "w");
    f.close();
}

// Returns ulong.max when the available disk space could not be determined.
ulong getAvailableDiskSpace(string path)
{
    import std.string: toStringz;
    version (Windows)
    {
        import std.path;
        import core.sys.windows.winbase;
        import core.sys.windows.winnt;
        import std.internal.cstring;

        ULARGE_INTEGER freeBytesAvailable;
        path ~= dirSeparator;
        auto success = GetDiskFreeSpaceExW(path.tempCStringW(), &freeBytesAvailable, null, null);
        return success ? freeBytesAvailable.QuadPart : ulong.max;
    }
    else
    {
        import core.sys.posix.sys.statvfs;

        statvfs_t stats;
        int err = statvfs(path.toStringz(), &stats);
        return !err ? stats.f_bavail * stats.f_frsize : ulong.max;
    }
}

struct CachePruner
{
    enum timestampFilename = "ircache_prune_timestamp";

    string cachePath; // absolute path
    Duration pruneInterval; // minimum time between pruning
    Duration expireDuration; // cache file expiration
    ulong sizeLimit; // in bytes
    uint sizeLimitPercentage; // Percentage limit of available space
    bool willPruneForSize; // true if we need to prune for absolute/relative size

    this(string cachePath, uint pruneIntervalSeconds, uint expireIntervalSeconds,
        ulong sizeLimit, uint sizeLimitPercentage)
    {
        import std.path;
        if (cachePath.isRooted())
            this.cachePath = cachePath.dup;
        else
            this.cachePath = absolutePath(expandTilde(cachePath));
        this.pruneInterval = dur!"seconds"(pruneIntervalSeconds);
        this.expireDuration = dur!"seconds"(expireIntervalSeconds);
        this.sizeLimit = sizeLimit;
        this.sizeLimitPercentage = sizeLimitPercentage < 100 ? sizeLimitPercentage : 100;
        this.willPruneForSize = (sizeLimit > 0) || (sizeLimitPercentage < 100);
    }

    void doPrune()
    {
        if (!exists(cachePath))
            return;

        if (!hasPruneIntervalPassed())
            return;

        // Only delete files that match LDC's cache file naming.
        // E.g.            "ircache_00a13b6f918d18f9f9de499fc661ec0d.o"
        auto filePattern = "ircache_????????????????????????????????.{o,obj}";
        auto cacheFiles = dirEntries(cachePath, filePattern, SpanMode.shallow, /+ followSymlink +/ false);

        // Delete all temporary files.
        deleteFiles(cachePath, filePattern ~ ".tmp???????");

        // Files that have not yet expired, may still be removed during pruning for size later.
        // This array holds the prune candidates after pruning for expiry.
        DirEntry[] pruneForSizeCandidates;
        ulong cacheSize;
        pruneForExpiry(cacheFiles, pruneForSizeCandidates, cacheSize);
        if (!willPruneForSize || !pruneForSizeCandidates.length)
            return;

        pruneForSize(pruneForSizeCandidates, cacheSize);
    }

private:
    void deleteFiles(string path, string filePattern)
    {
        foreach (DirEntry f; dirEntries(path, filePattern, SpanMode.shallow, /+ followSymlink +/ false))
        {
            try
            {
                remove(f.name);
            }
            catch (FileException)
            {
                // Simply skip the file when an error occurs.
                continue;
            }
        }
    }

    void pruneForExpiry(T)(T cacheFiles, out DirEntry[] remainingPruneCandidates, out ulong cacheSize)
    {
        foreach (DirEntry f; cacheFiles)
        {
            if (!f.isFile())
                continue;

            if (f.timeLastAccessed < (Clock.currTime - expireDuration))
            {
                try
                {
                    remove(f.name);
                }
                catch (FileException)
                {
                    // Simply skip the file when an error occurs.
                    continue;
                }
            }
            else if (willPruneForSize)
            {
                cacheSize += f.size;
                remainingPruneCandidates ~= f;
            }
        }
    }

    void pruneForSize(DirEntry[] candidates, ulong cacheSize)
    {
        ulong availableSpace = cacheSize + getAvailableDiskSpace(cachePath);
        if (!isSizeAboveMaximum(cacheSize, availableSpace))
            return;

        // Create heap ordered with most recently accessed files last.
        import std.container.binaryheap : heapify;
        auto candidateHeap = heapify!("a.timeLastAccessed > b.timeLastAccessed")(candidates);
        while (!candidateHeap.empty())
        {
            auto candidate = candidateHeap.front();
            candidateHeap.popFront();

            try
            {
                remove(candidate.name);
                // Update cache size
                cacheSize -= candidate.size;

                if (!isSizeAboveMaximum(cacheSize, availableSpace))
                    break;
            }
            catch (FileException)
            {
                // Simply skip the file when an error occurs.
            }
        }
    }

    // Checks if the prune interval has passed, and if so, creates/updates the pruning timestamp.
    bool hasPruneIntervalPassed()
    {
        import std.path: buildPath;
        auto fname = buildPath(cachePath, timestampFilename);
        if (pruneInterval == dur!"seconds"(0) || timeLastModified(fname,
                SysTime.min) < (Clock.currTime - pruneInterval))
        {
            writeEmptyFile(fname);
            return true;
        }
        return false;
    }

    bool isSizeAboveMaximum(ulong cacheSize, ulong availableSpace)
    {
        if (availableSpace == 0)
            return true;

        bool tooLarge = false;
        if (sizeLimit > 0)
            tooLarge = cacheSize > sizeLimit;

        tooLarge = tooLarge || ((100 * cacheSize) / availableSpace) > sizeLimitPercentage;

        return tooLarge;
    }
}
