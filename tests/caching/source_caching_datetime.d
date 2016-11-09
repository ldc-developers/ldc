// Clear the cache before running the tests (just in case this test was already run once).
// RUN: %ldc -c -cache=%T/sourcecache_dt %s -of=%t%obj \
// RUN: && %prunecache -f --max-bytes=1 %T/sourcecache_dt  \
// RUN: && %ldc -c -cache=%T/sourcecache_dt -cache-sourcefiles %s -of=%t%obj -vv -d-version=TIME      | FileCheck --check-prefix=NO_HIT %s \
// RUN: && %ldc -c -cache=%T/sourcecache_dt -cache-sourcefiles %s -of=%t%obj -vv -d-version=TIME      | FileCheck --check-prefix=NO_HIT %s \
// RUN: && %ldc -c -cache=%T/sourcecache_dt -cache-sourcefiles %s -of=%t%obj -vv -d-version=DATE      | FileCheck --check-prefix=NO_HIT %s \
// RUN: && %ldc -c -cache=%T/sourcecache_dt -cache-sourcefiles %s -of=%t%obj -vv -d-version=DATE      | FileCheck --check-prefix=NO_HIT %s \
// RUN: && %ldc -c -cache=%T/sourcecache_dt -cache-sourcefiles %s -of=%t%obj -vv -d-version=TIMESTAMP | FileCheck --check-prefix=NO_HIT %s \
// RUN: && %ldc -c -cache=%T/sourcecache_dt -cache-sourcefiles %s -of=%t%obj -vv -d-version=TIMESTAMP | FileCheck --check-prefix=NO_HIT %s

// NO_HIT: No cache manifest found for this build

void main()
{
    version(TIME)
        auto t = mixin("__TI" ~ "ME__");
    version(DATE)
        auto d = mixin("__DA" ~ "TE__");
    version(TIMESTAMP)
        auto ts = mixin("__TIM" ~ "ESTAMP__");
}
