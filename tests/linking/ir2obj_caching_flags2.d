// Test that certain cmdline flags result in different cache objects, even though the LLVM IR may be the same.
// Test a few fsanitize-coverage options.

// Note that the NO_HIT tests should change the default setting of the tested flag.

// Create and then empty the cache for correct testing when running the test multiple times.
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir
// RUN: %prunecache -f %t-dir --max-bytes=1
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -g                                 -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -fsanitize-coverage=trace-pc-guard -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -fsanitize-coverage=8bit-counters  -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -fsanitize-coverage=trace-cmp      -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -g                                 -vv | FileCheck --check-prefix=MUST_HIT %s
// The last test is a MUST_HIT test (hits with the first compile invocation), to make sure that the cache wasn't pruned somehow which could effectively disable some NO_HIT tests.

// MUST_HIT: Cache object found!
// NO_HIT: Cache object not found.

// Could hit is used for cases where we could have a cache hit, but currently we don't: a "TODO" item.
// COULD_HIT: Cache object

void main() {}
