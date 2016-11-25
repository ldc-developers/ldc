// Test recognition of -cache-retrieval commandline flag

// RUN: %ldc -c -of=%t%obj -cache=%T/cachedirectory %s -vv | FileCheck --check-prefix=FIRST %s \
// RUN: && %ldc -c -of=%t%obj -cache=%T/cachedirectory %s -cache-retrieval=copy -vv | FileCheck --check-prefix=MUST_HIT %s \
// RUN: && %ldc %t%obj \
// RUN: && %ldc -c -of=%t%obj -cache=%T/cachedirectory %s -cache-retrieval=link -vv | FileCheck --check-prefix=MUST_HIT %s \
// RUN: && %ldc %t%obj \
// RUN: && %ldc -c -of=%t%obj -cache=%T/cachedirectory %s -cache-retrieval=hardlink -vv | FileCheck --check-prefix=MUST_HIT %s \
// RUN: && %ldc %t%obj

// FIRST: Use IR-to-Object cache in {{.*}}cachedirectory
// Don't check whether the object is in the cache on the first run, because if this test is ran twice the cache will already be there.

// MUST_HIT: Use IR-to-Object cache in {{.*}}cachedirectory
// MUST_HIT: Cache object found!

void main()
{
}
