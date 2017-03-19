//===-- tools/ldc-prune-cache.d -----------------------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Prunes LDC's cache.
//
// TODO: Let the commandline parameters accept units, e.g.
//       `--interval=30m`, or `--max-bytes=5GB`.
//
//===----------------------------------------------------------------------===//

module ldc_prune_cache;

import std.stdio;
import std.getopt;
import std.file: isDir;

import driver.cache_pruning;

// System exit codes:
enum EX_OK = 0;
enum EX_USAGE = 64;

int main(string[] args)
{
    bool force, showHelp, error;
    uint pruneIntervalSeconds = 20 * 60;
    uint expireIntervalSeconds = 7 * 24 * 3600;
    ulong sizeLimitBytes = 0;
    uint sizeLimitPercentage = 75;

    try
    {
        getopt(args,
            "f|force", &force,
            "h|help", &showHelp,
            "interval", &pruneIntervalSeconds,
            "expiry", &expireIntervalSeconds,
            "max-bytes", &sizeLimitBytes,
            "max-percentage-of-avail", &sizeLimitPercentage
        );
    }
    catch(Exception e)
    {
        stderr.writeln(e.msg);
        stderr.writeln();
        args.length = 1; // Force display of help message.
    }

    if (showHelp || args.length != 2)
    {
        stderr.writef(q"EOS
OVERVIEW: LDC-PRUNE-CACHE
  Prunes the LDC's object file cache to prevent the cache growing infinitely.
  When a minimum pruning interval has passed (--interval), the following
  pruning scheme is executed:
  1. remove cached files that have passed the expiry duration (--expiry);
  2. remove cached files (oldest first) until the total cache size is below a
     set limit (--max-bytes, --max-percentage-of-avail).

USAGE: ldc-prune-cache [OPTION]... PATH
  PATH should be a directory where LDC has placed its object files cache (see
  LDC's -cache option).

OPTIONS:
  --expiration=<dur>     Sets the pruning expiration time of cache files to
                         <dur> seconds (default: 1 week).
  -f, --force            Force pruning, ignoring the prune interval.
  -h, --help             Show this message.
  --interval=<dur>       Sets the cache pruning interval to <dur> seconds
                         (default: 20 min). Set to 0 to force pruning, see -f.
  --max-bytes=<size>     Sets the cache size absolute limit to <size> bytes
                         (default: no absolute limit).
  --max-percentage-of-avail=<perc>
                         Sets the cache size limit to <perc> percent of the
                         available disk space (default 75%%).
EOS");
        return showHelp ? EX_OK : EX_USAGE;
    }

    string cacheDirectory = args[1];
    if (!isDir(cacheDirectory))
    {
        stderr.write("PATH must be a directory.");
        return EX_USAGE;
    }

    auto pruner = CachePruner(cacheDirectory,
        force ? 0 : pruneIntervalSeconds, expireIntervalSeconds, sizeLimitBytes, sizeLimitPercentage);

    pruner.doPrune();

    return EX_OK;
}
