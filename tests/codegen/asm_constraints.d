// RUN: not %ldc -c -w %s 2>&1 | FileCheck %s

void main () {
    import ldc.llvmasm : __asm;
    // CHECK: Error: __asm constraint argument is invalid
    __asm("", "][");
}
