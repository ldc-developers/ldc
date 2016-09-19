// Test recognition of -cache-sourcefiles commandline flag

// Clear the cache before running the tests (just in case this test was already run once).
// RUN: %ldc -c -cache=%T/sourcecache1 %s -of=%t%obj \
// RUN: && %prunecache -f --max-bytes=1 %T/sourcecache1  \
// RUN: && %ldc -c -cache=%T/sourcecache1 -cache-sourcefiles %s -of=%t%obj -vv | FileCheck --check-prefix=FIRST %s \
// RUN: && %ldc -c -cache=%T/sourcecache1 -cache-sourcefiles %s -of=%t%obj -vv | FileCheck --check-prefix=SECOND %s

// FIRST: Do source-cached build
// FIRST: No cache manifest found for this build
// FIRST: Write cache manifest

// SECOND: Do source-cached build
// SECOND: Cache manifest checks out
// SECOND: Recovering outputs from cache

void main()
{
}
