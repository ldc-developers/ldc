// Test cache pruning for size

// This test assumes that the `void main(){}` object file size is below 200_000 bytes and above 200_000/2,
// such that rebuilding with version(NEW_OBJ_FILE) will clear the cache of all but the latest object file.

// RUN: %ldc %s -cache=%t-dir
// RUN: %ldc %s -cache=%t-dir -cache-prune -cache-prune-interval=0 -d-version=SLEEP
// RUN: %ldc %s -cache=%t-dir -cache-prune -cache-prune-interval=0 -vv | FileCheck --check-prefix=MUST_HIT %s
// RUN: %ldc %s -cache=%t-dir -cache-prune -cache-prune-interval=0 -vv -d-version=NEW_OBJ_FILE | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -cache=%t-dir -cache-prune -cache-prune-interval=0 -vv | FileCheck --check-prefix=MUST_HIT %s
// RUN: %ldc -d-version=SLEEP -run %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -cache-prune-interval=0 -cache-prune-maxbytes=200000 -vv | FileCheck --check-prefix=MUST_HIT %s
// RUN: %ldc %t%obj
// RUN: %ldc %s -cache=%t-dir -d-version=SLEEP -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc -d-version=SLEEP -run %s
// RUN: %ldc %s -cache=%t-dir -cache-prune-interval=1 -cache-prune-maxbytes=200000 -d-version=NEW_OBJ_FILE
// RUN: %ldc %s -cache=%t-dir -cache-prune -cache-prune-interval=0 -vv | FileCheck --check-prefix=NO_HIT %s

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
        // Sleep for 4 seconds, so we are sure that the cache object file timestamps are "aging".
        import core.thread;
        Thread.sleep( dur!"seconds"(4) );
    }
}
