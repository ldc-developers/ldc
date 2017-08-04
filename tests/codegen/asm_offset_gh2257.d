// Test variable offsets in inline asm string

// Outputting to assembly (-output-s) ensures that LLVM accepts the final assembly code

// REQUIRES: target_X86

// TODO: it's not clear what the LLVM IR output should be exactly ("4+$0" or "4$0)...
// R UN: %ldc -betterC -mtriple=x86_64-linux-gnu -c -output-ll -of=%t.ll %s && FileCheck --check-prefix=LLVM %s < %t.ll

// RUN: %ldc -betterC -mtriple=x86_64-linux-gnu -c -output-s -of=%t.s -relocation-model=static  %s && FileCheck --check-prefix=ASM %s < %t.s
// RUN: %ldc -betterC -mtriple=x86_64-linux-gnu -c -output-s -of=%t.pic.s -relocation-model=pic %s && FileCheck --check-prefix=ASM %s < %t.pic.s

// RUN: %ldc -betterC --mtriple=x86_64-apple-macosx -c -output-s -of=%t.osx.s -relocation-model=static  %s && FileCheck --check-prefix=ASM %s < %t.osx.s
// RUN: %ldc -betterC --mtriple=x86_64-apple-macosx -c -output-s -of=%t.osx.pic.s -relocation-model=pic %s && FileCheck --check-prefix=ASM %s < %t.osx.pic.s

// LLVM-LABEL: define {{.*}}9foofoofoo
// ASM-LABEL: 9foofoofoo
void foofoofoo()
{
    align(64) uint a;
    asm pure nothrow @nogc {
        // LLVM: call void asm sideeffect
        // LLVM-SAME: movl %eax, $0
        // ASM: movl %eax, (%rsp)
        mov a, EAX;
        // LLVM-SAME: movl %ebx, 4$1
        // ASM-NEXT: movl %ebx, 4(%rsp)
        mov 4+a, EBX;
        // LLVM-SAME: movl %ecx, 8$2
        // ASM-NEXT: movl %ecx, 8(%rsp)
        mov a+8, ECX;
        // LLVM-SAME: movl %edx, -12$3
        // ASM-NEXT: movl %edx, -12(%rsp)
        mov a-12, EDX;
        // LLVM-SAME: movl %eax, -12$4
        // ASM-NEXT: movl %eax, -12(%rsp)
        mov -12+a, EAX;
    }
}

// LLVM-LABEL: define {{.*}}18offset_upon_offset
// ASM-LABEL: 18offset_upon_offset
void offset_upon_offset()
{
    uint a;
    asm pure nothrow @nogc {
        // LLVM: call void asm sideeffect
        // LLVM-SAME: movl %ebx, 4$0
        // ASM: movl %ebx, -{{[0-9]+}}(%r{{.*}}
        // ASM-NEXT: movl %ebx, -{{[0-9]+}}(%r{{.*}}
        // ASM-NEXT: movl %ebx, -{{[0-9]+}}(%r{{.*}}
        mov 4+a, EBX;
        mov 4+a, EBX;
        mov 4+a, EBX;
        // Several identical lines making sure we won't match on code elsewhere
    }
}


__gshared uint[2] constants = [0x0001_0203, 0x0001_0203];

// LLVM-LABEL: define {{.*}}8offconst
// ASM-LABEL: 8offconst
void offconst()
{
    asm pure nothrow @nogc {
        // LLVM: call void asm sideeffect
        // LLVM-SAME: movdqa 4$0, %xmm7
        movdqa XMM7, [constants + 4];
    }
}

// LLVM-LABEL: define {{.*}}7offfunc
// ASM-LABEL: 7offfunc
void offfunc()
{
    asm pure nothrow @nogc {
        // LLVM: call void asm sideeffect
        // LLVM-SAME: movdqa 4$0, %xmm7
        movdqa XMM0, [offfunc + 4];
    }
}
