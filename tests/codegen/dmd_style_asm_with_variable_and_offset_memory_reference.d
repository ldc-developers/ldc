// REQUIRES: target_X86

// RUN: %ldc -mtriple=i686-pc-windows-msvc -O -output-s -of=%t.s %s && FileCheck %s < %t.s

uint getHighHalfWithoutDisplacement(ulong value)
{
    asm
    {
        // CHECK: movl    4(%esp), %eax
        mov EAX, dword ptr [value + 4];
    }
}

uint getHighHalfWithDisplacement(uint value1, uint value2)
{
    asm
    {
        // CHECK: movl -2(%ebp), %eax
        mov EAX, word ptr 2[value1];
    }
}

extern(C) __gshared ulong someGlobalVariable;

uint getHighHalfOfGlobal(ulong value)
{
    asm
    {
        // CHECK: movl    ((4+(-8))+_someGlobalVariable)+8, %eax
        mov EAX, dword ptr [someGlobalVariable + 4];
    }
}

void foo()
{
    align(32) uint[4] a;
    asm pure nothrow @nogc
    {
        // CHECK: movl  %ebx, 4(%esp)
        mov a+4, EBX;
    }
}
