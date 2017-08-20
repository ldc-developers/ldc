// Test recognition of -cache commandline flag

// RUN: %ldc -cache=%t-dir %s -vv | FileCheck --check-prefix=FIRST  %s
// RUN: %ldc -cache=%t-dir %s -vv | FileCheck --check-prefix=SECOND %s


// FIRST: Use IR-to-Object cache in {{.*}}-dir
// Don't check whether the object is in the cache on the first run, because if this test is ran twice the cache will already be there.

// SECOND: Use IR-to-Object cache in {{.*}}-dir
// SECOND: Cache object found!

void main()
{
}
