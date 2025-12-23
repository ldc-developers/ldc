// Tests corner cases for naked functions with DMD-style inline asm.
//
// This tests:
// 1. Stack manipulation (push/pop)
// 2. Forward and backward jumps
// 3. Nested labels
// 4. Naked function calling convention

// REQUIRES: target_X86
// REQUIRES: host_X86

// FileCheck verification uses explicit triple for reproducible output
// RUN: %ldc -mtriple=x86_64-linux-gnu -O0 -output-s -of=%t.s %s
// RUN: FileCheck %s --check-prefix=ASM < %t.s

// Runtime verification uses native platform
// RUN: %ldc -O0 -run %s

module naked_asm_corner_cases;

// Test 1: Stack manipulation with push/pop
// ASM-LABEL: stackManipulation:
// ASM-NOT: pushq %rbp
// ASM-NOT: movq %rsp, %rbp
// ASM: pushq %rbx
// ASM: movl $42, %eax
// ASM: movl %eax, %ebx
// ASM: movl %ebx, %eax
// ASM: popq %rbx
// ASM: retq
extern(C) int stackManipulation() {
    asm { naked; }
    asm {
        push RBX;           // Save callee-saved register
        mov EAX, 42;
        mov EBX, EAX;       // Use the saved register
        mov EAX, EBX;
        pop RBX;            // Restore
        ret;
    }
}

// Test 2: Forward jump (jump to label defined later)
// ASM-LABEL: forwardJump:
// ASM: jmp .LforwardJump_skip
// ASM: .LforwardJump_skip:
// ASM: retq
extern(C) int forwardJump() {
    asm { naked; }
    asm {
        mov EAX, 1;
        jmp skip;           // Forward jump
        mov EAX, 0;         // Should be skipped
    skip:
        ret;
    }
}

// Test 3: Backward jump (loop)
// ASM-LABEL: backwardJump:
// ASM: .LbackwardJump_again:
// ASM: incl %eax
// ASM: cmpl $5, %eax
// ASM: jl .LbackwardJump_again
extern(C) int backwardJump() {
    asm { naked; }
    asm {
        xor EAX, EAX;
    again:
        inc EAX;
        cmp EAX, 5;
        jl again;           // Backward jump
        ret;
    }
}

// Test 4: Multiple control flow paths
// ASM-LABEL: multiPath:
// ASM: .LmultiPath_path1:
// ASM: .LmultiPath_path2:
// ASM: .LmultiPath_done:
extern(C) int multiPath(int x) {
    asm { naked; }
    version(D_InlineAsm_X86_64) asm {
        // x is in EDI on SysV ABI
        test EDI, EDI;
        jz path1;
        jmp path2;
    path1:
        mov EAX, 10;
        jmp done;
    path2:
        mov EAX, 20;
    done:
        ret;
    }
}

// Test 5: Naked function with static variable declaration (triggers Declaration_codegen)
// This tests that static declarations inside naked functions work correctly.
// The visitor's ExpStatement::visit calls Declaration_codegen for these,
// which requires an active IR insert point.
// ASM-LABEL: nakedWithStaticDecl:
// ASM: movl $42, %eax
// ASM: retq
extern(C) int nakedWithStaticDecl() {
    // Static variable declaration - triggers Declaration_codegen in visitor
    static immutable int staticVal = 42;
    asm { naked; }
    asm {
        mov EAX, 42;  // Use literal value since asm can't reference D variables
        ret;
    }
}

// Test 6: Runtime verification
void main() {
    // Verify stack manipulation works
    assert(stackManipulation() == 42, "stackManipulation failed");

    // Verify forward jump works
    assert(forwardJump() == 1, "forwardJump failed");

    // Verify backward jump (loop) works
    assert(backwardJump() == 5, "backwardJump failed");

    // Verify multi-path control flow
    assert(multiPath(0) == 10, "multiPath(0) failed");
    assert(multiPath(1) == 20, "multiPath(1) failed");

    // Verify naked function with static declaration works
    assert(nakedWithStaticDecl() == 42, "nakedWithStaticDecl failed");
}
