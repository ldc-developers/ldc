// Clear the cache before running the tests (just in case this test was already run once).
// RUN: %ldc -c -cache=%T/sourcecache_time %s -of=%t%obj \
// RUN: && %prunecache -f --max-bytes=1 %T/sourcecache_time  \
// RUN: && %ldc -c -cache=%T/sourcecache_time -cache-sourcefiles %s -of=%t%obj -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN: && %ldc -d-version=SLEEP -run %s \
// RUN: && %ldc -c -cache=%T/sourcecache_time -cache-sourcefiles %s -of=%t%obj -vv | FileCheck --check-prefix=NO_HIT %s

// NO_HIT: No cache manifest found for this build

void main()
{
    auto t = __TIME__;

    version (SLEEP)
    {
        // Sleep for 2 seconds, so we are certain that the __TIME__ timestamp has changed.
        import core.thread;
        Thread.sleep( dur!"seconds"(2) );
    }
}
