// ThinLTO: Test inline assembly functions with thinlto

// REQUIRES: LTO
// REQUIRES: host_X86

// Naked DMD-style asm (LLVM module-level inline assembly) with LTO on Windows
// is only supported since LLVM 9.
// REQUIRES: !Windows || atleast_llvm900

// RUN: %ldc -flto=thin %S/inputs/asm_x86.d -c -of=%t_input%obj
// RUN: %ldc -flto=thin -I%S %s %t_input%obj

import inputs.asm_x86;

int main() {
    return simplefunction();
}
