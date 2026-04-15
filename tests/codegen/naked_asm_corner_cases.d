// Tests corner cases for naked functions with DMD-style inline asm.
//
// This tests:
// 1. Stack manipulation (push/pop)
// 2. Forward and backward jumps
// 3. Nested labels
// 4. Naked function calling convention

// REQUIRES: host_X86 && target_X86

// RUN: %ldc -run %s

module naked_asm_corner_cases;

// Test 1: Stack manipulation with push/pop
extern(C) int stackManipulation() {
    asm { naked; }
    // Save callee-saved register
    version (D_InlineAsm_X86_64)
        asm { push RBX; }
    else
        asm { push EBX; }
    asm {
        mov EAX, 42;
        mov EBX, EAX;       // Use the saved register
        mov EAX, EBX;
    }
    // Restore
    version (D_InlineAsm_X86_64)
        asm { pop RBX; }
    else
        asm { pop EBX; }
    asm { ret; }
}

// Test 2: Forward jump (jump to label defined later)
extern(C) int forwardJump() {
    asm {
        naked;
        mov EAX, 1;
        jmp skip;           // Forward jump
        mov EAX, 0;         // Should be skipped
    skip:
        ret;
    }
}

// Test 3: Backward jump (loop)
extern(C) int backwardJump() {
    asm {
        naked;
        xor EAX, EAX;
    again:
        inc EAX;
        cmp EAX, 5;
        jl again;           // Backward jump
        ret;
    }
}

// Test 4: Multiple control flow paths
extern(C) int multiPath(int x) {
    asm { naked; }
    version (D_InlineAsm_X86_64) {
        // x is in ECX for the Win64 ABI, otherwise EDI for SysV
        version (Windows)
            asm { test ECX, ECX; }
        else
            asm { test EDI, EDI; }
    } else {
        // x is on stack at [ESP+4] for 32-bit cdecl
        asm {
            mov EAX, [ESP+4];
            test EAX, EAX;
        }
    }
    asm {
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

// Test 5: Naked function with static immutable variable declaration
extern(C) int nakedWithStaticDecl() {
    static immutable int staticVal = 42;
    asm {
        naked;
        mov EAX, staticVal;
        ret;
    }
}

void main() {
    assert(stackManipulation() == 42, "stackManipulation failed");
    assert(forwardJump() == 1, "forwardJump failed");
    assert(backwardJump() == 5, "backwardJump failed");
    assert(multiPath(0) == 10, "multiPath(0) failed");
    assert(multiPath(1) == 20, "multiPath(1) failed");
    assert(nakedWithStaticDecl() == 42, "nakedWithStaticDecl failed");
}
