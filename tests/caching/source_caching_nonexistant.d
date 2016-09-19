// Test that caching works correctly if new files appear compared to the cached build (new files appear earlier in the search path).

// The first few RUN commands make sure that the testrun starts with a clean slate:
// 1. An empty cache.
// 2. testdir/exist.d does not exist.

// RUN: %ldc -c -cache=%T/exist1 %s -I%T -I%S/inputs/nonexist -of=%t%obj \
// RUN: && %prunecache -f --max-bytes=1 %T/exist1  \
// RUN: && removefile %T/exist.d \
// RUN: && %ldc -c -cache=%T/exist1 -cache-sourcefiles %s -I%T -I%S/inputs/nonexist -of=%t%obj -vv | FileCheck --check-prefix=FIRST %s \
// RUN: && copyfile %S/inputs/nonexist/exist.d %T/exist.d \
// RUN: && %ldc -c -cache=%T/exist1 -cache-sourcefiles %s -I%T -I%S/inputs/nonexist -of=%t%obj -vv | FileCheck --check-prefix=SECOND %s

// FIRST: Do source-cached build
// FIRST: No cache manifest found for this build
// FIRST: Write cache manifest

// SECOND: Do source-cached build
// SECOND: Cache manifest fail: previously non-existant file now exists
// SECOND: Write cache manifest

import exist;

void main()
{
}
