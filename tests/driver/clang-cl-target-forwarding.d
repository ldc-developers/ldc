// On Windows, either clang-cl or cl is used to preprocess C.
// If clang-cl is used, LDC's target triple and any target features should be
// passed to clang-cl.

// REQUIRES: Windows && target_X86 && target_AArch64

// RUN: %ldc -mtriple=x86_64-pc-windows-msvc -mcpu=znver1 -v -c %S/inputs/preprocessable.c | FileCheck %s -check-prefix=znver1
// RUN: %ldc -mtriple=x86_64-pc-windows-msvc -mcpu=znver1 -mattr -sse4a -v -c %S/inputs/preprocessable.c | FileCheck %s -check-prefix=znver1-sans-sse4a
// RUN: %ldc -mtriple=x86_64-pc-windows-msvc -mcpu=znver2 -v -c %S/inputs/preprocessable.c | FileCheck %s -check-prefix=znver2
// RUN: %ldc -mtriple=aarch64-pc-windows-msvc -mcpu=apple-a10 -v -c %S/inputs/preprocessable.c | FileCheck %s -check-prefix=apple-a10
// RUN: %ldc -mtriple=aarch64-pc-windows-msvc -mcpu=apple-a11 -v -c %S/inputs/preprocessable.c | FileCheck %s -check-prefix=apple-a11

// znver1: {{\\cl\.exe[[:space:]]|\\clang-cl\.exe[[:space:]].*-target[[:space:]]+x86_64-pc-windows-msvc.*-Xclang[[:space:]]+-target-feature[[:space:]]+-Xclang[[:space:]]+\-clwb.*-Xclang[[:space:]]+-target-feature[[:space:]]+-Xclang[[:space:]]+\+sse4a}}
// znver1-sans-sse4a: {{\\cl\.exe[[:space:]]|\\clang-cl\.exe[[:space:]].*-target[[:space:]]+x86_64-pc-windows-msvc.*-Xclang[[:space:]]+-target-feature[[:space:]]+-Xclang[[:space:]]+\-clwb.*-Xclang[[:space:]]+-target-feature[[:space:]]+-Xclang[[:space:]]+\-sse4a}}
// znver2: {{\\cl\.exe[[:space:]]|\\clang-cl\.exe[[:space:]].*-target[[:space:]]+x86_64-pc-windows-msvc.*-Xclang[[:space:]]+-target-feature[[:space:]]+-Xclang[[:space:]]+\+clwb}}
// apple-a10: {{\\cl\.exe[[:space:]]|\\clang-cl\.exe[[:space:]].*-target[[:space:]]+aarch64-pc-windows-msvc.*-Xclang[[:space:]]+-target-feature[[:space:]]+-Xclang[[:space:]]+\+apple-a10.*-Xclang[[:space:]]+-target-feature[[:space:]]+-Xclang[[:space:]]+\-apple-a11}}
// apple-a11: {{\\cl\.exe[[:space:]]|\\clang-cl\.exe[[:space:]].*-target[[:space:]]+aarch64-pc-windows-msvc.*-Xclang[[:space:]]+-target-feature[[:space:]]+-Xclang[[:space:]]+\-apple-a10.*-Xclang[[:space:]]+-target-feature[[:space:]]+-Xclang[[:space:]]+\+apple-a11}}
