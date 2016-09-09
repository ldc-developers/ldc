// Test ir2obj-cache pruning for size

// REQUIRES: atleast_llvm309

// This test assumes that the `void main(){}` object file size is below and somewhat close to 2000 bytes,
// such that rebuilding with version(NEW_OBJ_FILE) will clear the cache of all but the latest object file.

// RUN: %ldc %s -ir2obj-cache=%T/cachedirectory \
// RUN: && %ldc %s -ir2obj-cache=%T/cachedirectory -ir2obj-cache-prune -ir2obj-cache-prune-interval=0 -d-version=SLEEP \
// RUN: && %ldc %s -ir2obj-cache=%T/cachedirectory -ir2obj-cache-prune -ir2obj-cache-prune-interval=0 -vv | FileCheck --check-prefix=MUST_HIT %s \
// RUN: && %ldc %s -ir2obj-cache=%T/cachedirectory -ir2obj-cache-prune -ir2obj-cache-prune-interval=0 -vv -d-version=NEW_OBJ_FILE | FileCheck --check-prefix=NO_HIT %s \
// RUN: && %ldc %s -ir2obj-cache=%T/cachedirectory -ir2obj-cache-prune -ir2obj-cache-prune-interval=0 -vv | FileCheck --check-prefix=MUST_HIT %s \
// RUN: && %ldc -d-version=SLEEP -run %s \
// RUN: && %ldc %s -ir2obj-cache=%T/cachedirectory -ir2obj-cache-prune -ir2obj-cache-prune-interval=0 -ir2obj-cache-prune-maxbytes=2000 -vv | FileCheck --check-prefix=MUST_HIT %s \
// RUN: && %ldc %s -ir2obj-cache=%T/cachedirectory -d-version=SLEEP -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN: && %ldc -d-version=SLEEP -run %s \
// RUN: && %ldc %s -ir2obj-cache=%T/cachedirectory -ir2obj-cache-prune -ir2obj-cache-prune-interval=0 -ir2obj-cache-prune-maxbytes=2000 -d-version=NEW_OBJ_FILE \
// RUN: && %ldc %s -ir2obj-cache=%T/cachedirectory -ir2obj-cache-prune -ir2obj-cache-prune-interval=0 -vv | FileCheck --check-prefix=NO_HIT %s

// MUST_HIT: Cache object found!
// NO_HIT-NOT: Cache object found!

void main()
{
    version(NEW_OBJ_FILE)
    {
        auto a = __TIME__;
    }

    version(SLEEP)
    {
        // Sleep for 2 seconds, so we are sure that the cache object file timestamps are "aging".
        import core.thread;
        Thread.sleep( dur!("seconds")( 2 ) );
    }
}
