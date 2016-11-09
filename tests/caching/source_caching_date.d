// Test that using __DATE__ does not automatically invalidate caching.
// Caveat: this test fails when a date change happens in-between the two tested compile runs.

// Clear the cache before running the tests (just in case this test was already run once).
// RUN: %ldc -c -cache=%T/sourcecache_date %s -of=%t%obj \
// RUN: && %prunecache -f --max-bytes=1 %T/sourcecache_date  \
// RUN: && %ldc -c -cache=%T/sourcecache_date -cache-sourcefiles %s -of=%t%obj -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN: && %ldc -c -cache=%T/sourcecache_date -cache-sourcefiles %s -of=%t%obj -vv | FileCheck --check-prefix=HIT %s

// NO_HIT: No cache manifest found for this build
// HIT: Cache manifest checks out
// HIT: Recovering outputs from cache

void main()
{
    auto t = __DATE__;
}
