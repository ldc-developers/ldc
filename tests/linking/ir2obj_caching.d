// Test recognition of -ir2obj-cache commandline flag

// RUN: %ldc -ir2obj-cache=%T/cachedirectory %s -vv | FileCheck --check-prefix=FIRST %s \
// RUN: && %ldc -ir2obj-cache=%T/cachedirectory %s -vv | FileCheck --check-prefix=SECOND %s


// FIRST: Use IR-to-Object cache in {{.*}}cachedirectory
// Don't check whether the object is in the cache on the first run, because if this test is ran twice the cache will already be there.

// SECOND: Use IR-to-Object cache in {{.*}}cachedirectory
// SECOND: Cache object found!
// SECOND: SymLink output to cached object file

void main()
{
}
