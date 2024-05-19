// Test that llvm.used is emitted correctly when multiple D modules are compiled into one LLVM module.

// REQUIRES: target_X86

// Explicitly use OS X triple, so that llvm.used is used for moduleinfo globals.
// RUN: %ldc -c -output-ll -O3 %S/inputs/module_ctor.d %s -of=%t.ll -mtriple=x86_64-apple-macosx && FileCheck --check-prefix=LLVM %s < %t.ll

// RUN: %ldc -O3 %S/inputs/module_ctor.d -run %s | FileCheck --check-prefix=EXECUTE %s

// There was a bug where llvm.used was emitted more than once, whose symptom was that suffixed versions would appear: e.g. `@llvm.used.3`.
// Expect 2 llvm.used entries, for both ModuleInfo refs.
// LLVM-NOT: @llvm.used.
// LLVM: @llvm.used = appending global [2 x ptr]
// LLVM-NOT: @llvm.used.

// EXECUTE: ctor
// EXECUTE: main
// EXECUTE: dtor

import core.stdc.stdio;

static ~this()
{
    puts("dtor\n");
}

void main() {
    puts("main\n");
}
