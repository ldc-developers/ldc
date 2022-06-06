// Test value name discarding in conjunction with the compile cache: local variable name changes should still give a cache hit.

// Create and then empty the cache for correct testing when running the test multiple times.
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir
// RUN: %prunecache -f %t-dir --max-bytes=1
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -d-version=FIRST -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -vv | FileCheck --check-prefix=MUST_HIT %s

// MUST_HIT: Cache object found!
// NO_HIT-NOT: Cache object found!

version (FIRST)
{
    int foo(int a)
    {
        return a + 2;
    }
}
else
{
    int foo(int differentname)
    {
        return differentname + 2;
    }
}
