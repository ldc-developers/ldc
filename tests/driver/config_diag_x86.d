// REQUIRES: target_X86

// RUN: %ldc -conf=%S/inputs/override_default.conf -mtriple=x86-apple-windows-msvc -c -o- %s | FileCheck %s --check-prefix=OVERRIDE_DEFAULT
// OVERRIDE_DEFAULT: LDC - the LLVM D compiler


void foo()
{
}
