// Test that certain cmdline flags result in different ir2obj cache objects, even though the LLVM IR may be the same.

// Note that the NO_HIT tests should change the default setting of the tested flag.

// Create and then empty the cache for correct testing when running the test multiple times.
// RUN: %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache \
// RUN:   && %prunecache -f %T/flag1cache --max-bytes=1 \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -g                               -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache                                  -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -O                               -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -O3                              -vv | FileCheck --check-prefix=MUST_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -O2                              -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -O4                              -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc -O5 %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache                              -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -Os                              -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -Oz                              -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -disable-d-passes                -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -disable-simplify-drtcalls       -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -disable-simplify-libcalls       -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -disable-gc2stack                -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -enable-inlining                 -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -unit-at-a-time=false            -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -strip-debug                     -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -disable-loop-unrolling          -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -disable-loop-vectorization      -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -disable-slp-vectorization       -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -vectorize-loops                 -vv | FileCheck --check-prefix=NO_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -v -wi -d                        -vv | FileCheck --check-prefix=MUST_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -D -H -I. -J.                    -vv | FileCheck --check-prefix=MUST_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -d-version=Irrelevant            -vv | FileCheck --check-prefix=MUST_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -unittest                        -vv | FileCheck --check-prefix=MUST_HIT %s \
// RUN:   && %ldc %s               -ir2obj-cache=%T/flag1cache -lib                             -vv | FileCheck --check-prefix=MUST_HIT %s \
// RUN:   && %ldc                  -ir2obj-cache=%T/flag1cache -vv -run %s                          | FileCheck --check-prefix=COULD_HIT %s \
// RUN:   && %ldc                  -ir2obj-cache=%T/flag1cache -vv -run %s a b                      | FileCheck --check-prefix=MUST_HIT %s \
// RUN:   && %ldc %s -c -of=%t%obj -ir2obj-cache=%T/flag1cache -g                               -vv | FileCheck --check-prefix=MUST_HIT %s
// The last test is a MUST_HIT test (hits with the first compile invocation), to make sure that the cache wasn't pruned somehow which could effectively disable some NO_HIT tests.

// MUST_HIT: Cache object found!
// NO_HIT: Cache object not found.

// Could hit is used for cases where we could have a cache hit, but currently we don't: a "TODO" item.
// COULD_HIT: Cache object

void main() {}
