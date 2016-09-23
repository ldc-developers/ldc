// Test value name discarding in conjunction with the ir2obj cache: local variable name changes should still give a cache hit.

// REQUIRES: atleast_llvm309

// Create and then empty the cache for correct testing when running the test multiple times.
// RUN: %ldc %s -c -of=%t%obj -ir2obj-cache=%T/dvni2oc \
// RUN:   && %prunecache -f %T/dvni2oc --max-bytes=1 \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/dvni2oc -d-version=FIRST -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/dvni2oc -vv | FileCheck --check-prefix=MUST_HIT %s

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
