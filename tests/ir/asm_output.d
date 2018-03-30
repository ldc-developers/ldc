// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s --check-prefix LLVM < %t.ll
// RUN: %ldc -c -output-s  -of=%t.s %s && FileCheck %s --check-prefix ASM < %t.s

// Try to keep these very simple checks independent of architecture.

// ASM: D7example3fooFZi:
int foo() {
// LLVM:  ret i32 42
    return 42;
}
