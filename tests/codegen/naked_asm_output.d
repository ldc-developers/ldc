// Tests that naked functions with DMD-style inline asm generate correct
// machine code without prologue/epilogue overhead.
//
// This verifies:
// 1. No function prologue (no push rbp, mov rbp rsp, etc.)
// 2. No function epilogue (no pop rbp before our explicit ret)
// 3. Labels are generated correctly
// 4. Template instantiations use proper linkage (comdat)

// REQUIRES: target_X86

// Test 1: Basic naked function - verify no prologue/epilogue
// RUN: %ldc -mtriple=x86_64-linux-gnu -O0 -output-s -of=%t.s %s
// RUN: FileCheck %s --check-prefix=ASM < %t.s

// Test 2: Verify LLVM IR has correct attributes
// RUN: %ldc -mtriple=x86_64-linux-gnu -O0 -output-ll -of=%t.ll %s
// RUN: FileCheck %s --check-prefix=IR < %t.ll

module naked_asm_output;

// ASM-LABEL: simpleNaked:
// ASM-NOT: pushq %rbp
// ASM-NOT: movq %rsp, %rbp
// ASM: xorl %eax, %eax
// ASM: retq
// ASM-NOT: popq %rbp

// IR-LABEL: define i32 @simpleNaked()
// IR-SAME: #[[ATTRS:[0-9]+]]
extern(C) int simpleNaked() {
    asm { naked; }
    asm {
        xor EAX, EAX;
        ret;
    }
}

// ASM-LABEL: nakedWithLabels:
// ASM-NOT: pushq %rbp
// ASM: xorl %eax, %eax
// ASM: .LnakedWithLabels_loop:
// ASM: incl %eax
// ASM: cmpl $10, %eax
// ASM: jl .LnakedWithLabels_loop
// ASM: retq

extern(C) int nakedWithLabels() {
    asm { naked; }
    asm {
        xor EAX, EAX;
    loop:
        inc EAX;
        cmp EAX, 10;
        jl loop;
        ret;
    }
}

// ASM-LABEL: nakedWithMultipleLabels:
// ASM-NOT: pushq %rbp
// ASM: .LnakedWithMultipleLabels_start:
// ASM: .LnakedWithMultipleLabels_middle:
// ASM: .LnakedWithMultipleLabels_end:
// ASM: retq

extern(C) int nakedWithMultipleLabels() {
    asm { naked; }
    asm {
        xor EAX, EAX;
    start:
        inc EAX;
        cmp EAX, 5;
        jl start;
    middle:
        inc EAX;
        cmp EAX, 10;
        jl middle;
    end:
        ret;
    }
}

// Template function - should have comdat for deduplication
// ASM: .section .text._D16naked_asm_output__T13nakedTemplateVii42ZQvFZi,"axG",@progbits,_D16naked_asm_output__T13nakedTemplateVii42ZQvFZi,comdat
// ASM-LABEL: _D16naked_asm_output__T13nakedTemplateVii42ZQvFZi:
// ASM-NOT: pushq %rbp
// ASM: movl $42, %eax
// ASM: retq

// IR-LABEL: define i32 @_D16naked_asm_output__T13nakedTemplateVii42ZQvFZi()
// IR-SAME: comdat

int nakedTemplate(int N)() {
    asm { naked; }
    asm {
        mov EAX, N;
        ret;
    }
}

// Instantiate template
int instantiate1() {
    return nakedTemplate!42();
}

// Verify naked attribute is present in attributes group
// IR: attributes #[[ATTRS]] = {{{.*}}naked{{.*}}noinline{{.*}}nounwind{{.*}}optnone
