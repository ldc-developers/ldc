// Test that inlining does not leak definitions without marking them as available_externally
// "Leaking" = symbols definitions in .o file that shouldn't be declarations instead (undefined symbols).

// REQUIRES: target_X86

// RUN: %ldc %s -mtriple=x86_64-linux-gnu -I%S -c -output-ll -release                  -O3 -enable-cross-module-inlining -of=%t.O3.ll && FileCheck %s --check-prefix OPT3 < %t.O3.ll
// RUN: %ldc %s -mtriple=x86_64-linux-gnu -I%S -c -output-ll -release -enable-inlining -O0 -enable-cross-module-inlining -of=%t.O0.ll && FileCheck %s --check-prefix OPT0 < %t.O0.ll

import inputs.inlinables_asm;

extern (C): // simplify mangling for easier matching

// Inlined naked asm func could end up as global symbols, definitely bad!
// (would give multiple definition linker error)
// OPT0-NOT: module asm {{.*}}.globl{{.*}}_naked_asm_func
// OPT3-NOT: module asm {{.*}}.globl{{.*}}_naked_asm_func

// OPT0-LABEL: define{{.*}} @asm_func(
// OPT3-LABEL: define{{.*}} @asm_func(
void asm_func()
{
    naked_asm_func();
    // OPT0: ret void
    // OPT3: ret void
}

// OPT0-LABEL: define{{.*}} @main(
// OPT3-LABEL: define{{.*}} @main(
int main()
{
    asm_func();

    return 0;
    // OPT0: ret i32 0
    // OPT3: ret i32 0
}
