// ThinLTO: Test inline assembly functions with thinlto

// REQUIRES: LTO
// REQUIRES: host_X86

// RUN: %ldc -flto=thin %S/inputs/asm_x86.d -c -of=%t_input%obj
// RUN: %ldc -flto=thin -I%S %s %t_input%obj

import inputs.asm_x86;

int main() {
    return simplefunction();
}
