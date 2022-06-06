// REQUIRES: target_X86

// RUN: %ldc -output-s -x86-asm-syntax=intel -mtriple=x86_64-linux-gnu -of=%t.s %s
// RUN: FileCheck %s < %t.s

// CHECK: _D17dmd_inline_asm_ip3fooFZm
ulong foo()
{
    asm
    {
        // CHECK: mov eax, dword ptr [eip]
        mov EAX, [EIP];
        // CHECK-NEXT: mov rax, qword ptr [rip]
        mov RAX, [RIP];
        ret;
    }
}
