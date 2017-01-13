// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s --check-prefix LLVM < %t.ll
// RUN: %ldc -c -output-s  -of=%t.s %s && FileCheck %s --check-prefix ASM < %t.s

int foo() {
    return 42;
// Try to keep these very simple checks independent of architecture:
// LLVM:  ret i32 42
// ASM:  {{(\$|#)}}42
}
