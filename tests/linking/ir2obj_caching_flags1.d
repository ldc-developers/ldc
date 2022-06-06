// Test that certain cmdline flags result in different cache objects, even though the LLVM IR may be the same.

// Note that the NO_HIT tests should change the default setting of the tested flag.

// Create and then empty the cache for correct testing when running the test multiple times.
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir
// RUN: %prunecache -f %t-dir --max-bytes=1
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -g                               -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir                                  -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -O                               -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -O3                              -vv | FileCheck --check-prefix=MUST_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -O2                              -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -O4                              -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc -O5 %s -c -of=%t%obj -cache=%t-dir                              -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -Os                              -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -Oz                              -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -disable-d-passes                -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -disable-simplify-drtcalls       -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -disable-simplify-libcalls       -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -disable-gc2stack                -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -enable-inlining                 -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -strip-debug                     -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -disable-loop-unrolling          -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -disable-loop-vectorization      -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -disable-slp-vectorization       -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -vectorize-loops                 -vv | FileCheck --check-prefix=NO_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -v -wi -d                        -vv | FileCheck --check-prefix=MUST_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -D -H -I. -J.                    -vv | FileCheck --check-prefix=MUST_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -d-version=Irrelevant            -vv | FileCheck --check-prefix=MUST_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -unittest                        -vv | FileCheck --check-prefix=MUST_HIT %s
// RUN: %ldc %s               -cache=%t-dir -lib                             -vv | FileCheck --check-prefix=MUST_HIT %s
// RUN: %ldc                  -cache=%t-dir -vv -run %s                          | FileCheck --check-prefix=COULD_HIT %s
// RUN: %ldc                  -cache=%t-dir -vv -run %s a b                      | FileCheck --check-prefix=MUST_HIT %s
// RUN: %ldc %s -c -of=%t%obj -cache=%t-dir -g                               -vv | FileCheck --check-prefix=MUST_HIT %s
// The last test is a MUST_HIT test (hits with the first compile invocation), to make sure that the cache wasn't pruned somehow which could effectively disable some NO_HIT tests.

// MUST_HIT: Cache object found!
// NO_HIT: Cache object not found.

// Could hit is used for cases where we could have a cache hit, but currently we don't: a "TODO" item.
// COULD_HIT: Cache object

void main() {}
