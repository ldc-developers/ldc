// REQUIRES: target_AArch64
// RUN: %ldc -mtriple=arm64-apple-ios -c -o- %s
// RUN: %ldc -mtriple=arm64-apple-tvos -c -o- %s
// RUN: %ldc -mtriple=arm64-apple-watchos -c -o- %s
// RUN: %ldc -mtriple=arm64-apple-xros -c -o- %s

version (Apple) {} else static assert(0, "version(Apple) missing");
