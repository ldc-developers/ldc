// Test ldc-cache-prune tool

// This test assumes that the `void main(){}` object file size is below 200_000 bytes and above 200_000/2,
// such that rebuilding with version(NEW_OBJ_FILE) will clear the cache of all but the latest object file.

// RUN: %ldc %s -cache=%t-dir
// RUN: %ldc %s -cache=%t-dir -d-version=SLEEP
// RUN: %prunecache -f %t-dir
// RUN: %ldc %s -cache=%t-dir -vv | FileCheck --check-prefix=MUST_HIT %s
// RUN: %ldc %s -cache=%t-dir -vv -d-version=NEW_OBJ_FILE | FileCheck --check-prefix=NO_HIT %s
// RUN: %prunecache %t-dir -f
// RUN: %ldc %s -cache=%t-dir -vv | FileCheck --check-prefix=MUST_HIT %s
// RUN: %ldc -d-version=SLEEP -run %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -vv | FileCheck --check-prefix=MUST_HIT %s
// RUN: %prunecache --force --max-bytes=200000 %t-dir
// RUN: %ldc %t%obj
// RUN: %ldc %s -cache=%t-dir -d-version=SLEEP -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc -d-version=SLEEP -run %s
// RUN: %ldc %s -cache=%t-dir -d-version=NEW_OBJ_FILE
// RUN: %prunecache --interval=0 %t-dir --max-bytes=200000
// RUN: %ldc %s -cache=%t-dir -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc -d-version=SLEEP -run %s
// RUN: %ldc -d-version=SLEEP -run %s
// RUN: %prunecache %t-dir -f --expiry=2
// RUN: %ldc %s -cache=%t-dir -vv | FileCheck --check-prefix=NO_HIT %s

// MUST_HIT: Cache object found!
// NO_HIT-NOT: Cache object found!

void main()
{
    // Add non-zero static data to guarantee a binary size larger than 200_000/2.
    static byte[120_000] dummy = 1;

    version (NEW_OBJ_FILE)
    {
        auto a = __TIME__;
    }

    version (SLEEP)
    {
        // Sleep for 2 seconds, so we are sure that the cache object file timestamps are "aging".
        import core.thread;
        Thread.sleep( dur!"seconds"(2) );
    }
}
