// Tests that GCC-style asm %= (unique ID) is correctly translated to LLVM ${:uid}
// which generates unique labels for each asm statement.
// See: https://github.com/ldc-developers/ldc/issues/4294

// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64-linux-gnu -output-s -of=%t.s %s
// RUN: FileCheck %s < %t.s

// Two functions with identical asm that use %= should get different label numbers.
// The ${:uid} mechanism in LLVM generates sequential numbers (0, 1, 2, ...).

int func1() {
    int result;
    // CHECK-LABEL: func1
    // CHECK: loop_[[ID1:[0-9]+]]:
    // CHECK: jmp loop_[[ID1]]
    asm {
        "xorl %0, %0\n\tloop_%=:\n\tincl %0\n\tjmp loop_%="
        : "=r" (result);
    }
    return result;
}

int func2() {
    int result;
    // CHECK-LABEL: func2
    // CHECK: loop_[[ID2:[0-9]+]]:
    // CHECK: jmp loop_[[ID2]]
    asm {
        "xorl %0, %0\n\tloop_%=:\n\tincl %0\n\tjmp loop_%="
        : "=r" (result);
    }
    return result;
}
