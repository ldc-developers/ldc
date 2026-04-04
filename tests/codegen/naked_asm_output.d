// Tests that naked functions with DMD-style inline asm generate correct
// machine code without prologue/epilogue overhead.
//
// This verifies:
// 1. No function prologue (no push rbp, mov rbp rsp, etc.)
// 2. No function epilogue (no pop rbp before our explicit ret)
// 3. Labels are generated correctly
// 4. Template instantiations use proper linkage (comdat)

// REQUIRES: target_X86

// Generate both .s (asm) and .ll (IR) outputs at once.
// RUN: %ldc -mtriple=x86_64-linux-gnu -O0 -output-s -output-ll -of=%t.s %s

// Test 1: Basic naked function - verify no prologue/epilogue
// RUN: FileCheck %s --check-prefix=ASM < %t.s

// Test 2: Verify LLVM IR has correct attributes
// RUN: FileCheck %s --check-prefix=IR < %t.ll

module naked_asm_output;

// ASM-LABEL: simpleNaked:
// ASM-NEXT: .cfi_startproc
// ASM-NEXT: #APP
// ASM-NEXT: xorl %eax, %eax
// ASM-NEXT: retq

// IR-LABEL: define i32 @simpleNaked()
// IR-SAME: #[[ATTRS:[0-9]+]]
extern(C) int simpleNaked() {
    asm {
        naked;
        xor EAX, EAX;
        ret;
    }
}

// ASM-LABEL: nakedWithLabels:
// ASM-NEXT: .cfi_startproc
// ASM-NEXT: #APP
// ASM-NEXT: xorl %eax, %eax
// ASM-NEXT: .LnakedWithLabels_loop:
// ASM-NEXT: incl %eax
// ASM-NEXT: cmpl $10, %eax
// ASM-NEXT: jl .LnakedWithLabels_loop
// ASM-NEXT: retq

extern(C) int nakedWithLabels() {
    asm {
        naked;
        xor EAX, EAX;
    loop:
        inc EAX;
        cmp EAX, 10;
        jl loop;
        ret;
    }
}

// ASM-LABEL: nakedWithMultipleLabels:
// ASM-NEXT: .cfi_startproc
// ASM-NEXT: #APP
// ASM-NEXT: jl .LnakedWithMultipleLabels_innerAsmLabel
// ASM-NEXT: .LnakedWithMultipleLabels_innerAsmLabel:
// ASM-NEXT: jl .LnakedWithMultipleLabels_otherAsmLabel
// ASM-NEXT: #NO_APP
// ASM-NEXT: #APP
// ASM-NEXT: .LnakedWithMultipleLabels_otherAsmLabel:
// ASM-NEXT: jl .LnakedWithMultipleLabels_dLabel
// ASM-NEXT: #NO_APP
// ASM-NEXT: #APP
// ASM-NEXT: .LnakedWithMultipleLabels_dLabel:
// ASM-NEXT: #NO_APP
// ASM-NEXT: #APP
// ASM-NEXT: jl .LnakedWithMultipleLabels_innerAsmLabel
// ASM-NEXT: jl .LnakedWithMultipleLabels_otherAsmLabel
// ASM-NEXT: retq

extern(C) int nakedWithMultipleLabels() {
    asm {
        naked;
        jl innerAsmLabel;
    innerAsmLabel:
        jl otherAsmLabel;
    }
    asm {
    otherAsmLabel:
        jl dLabel;
    }
dLabel:
    asm {
        jl innerAsmLabel;
        jl otherAsmLabel;
        ret;
    }
}

// Template function - should have comdat for deduplication
// ASM: .section .text._D16naked_asm_output__T13nakedTemplateVii42ZQvFZi,"axG",@progbits,_D16naked_asm_output__T13nakedTemplateVii42ZQvFZi,comdat
// ASM-LABEL: _D16naked_asm_output__T13nakedTemplateVii42ZQvFZi:
// ASM-NEXT: .cfi_startproc
// ASM-NEXT: #APP
// ASM-NEXT: movl $42, %eax
// ASM-NEXT: retq

// Template function check - use IR-DAG to allow flexible ordering since
// the template may be emitted after its caller (instantiate1)
// IR-DAG: define weak_odr i32 @_D16naked_asm_output__T13nakedTemplateVii42ZQvFZi() #[[ATTRS]] comdat
int nakedTemplate(int N)() {
    asm {
        naked;
        mov EAX, N;
        ret;
    }
}

// Instantiate template
int instantiate1() {
    return nakedTemplate!42();
}

// Verify naked and noinline attributes are present in attributes group
// IR: attributes #[[ATTRS]] = {{{.*}}naked{{.*}}noinline
