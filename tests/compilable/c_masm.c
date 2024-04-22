// DMD supports case-insensitive register operands for x86 assembly in C files, when targeting Windows.
// Additionally, a new-line can be used to terminate an instruction.

// REQUIRES: Windows && target_X86
// RUN: %ldc -mtriple=i686-pc-windows-msvc -c %s

unsigned int subtract(unsigned int a, unsigned int b) {
    __asm {
        mov eax, dword ptr [a]
        mov eDX, dword ptr [b]
        sub Eax, edx
    }
}
