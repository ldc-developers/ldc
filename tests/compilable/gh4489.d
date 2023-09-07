// REQUIRES: target_AArch64
// RUN: %ldc -mtriple=arm64-apple-macos -c %s

void foo(inout float[3]);

void bar()
{
    float[3] x;
    foo(x);
}
