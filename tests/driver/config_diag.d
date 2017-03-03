// RUN: not %ldc -conf=%S/inputs/noswitches.conf %s 2>&1 | FileCheck %s --check-prefix=NOSWITCHES
// NOSWITCHES: Could not look up switches in {{.*}}noswitches.conf

// RUN: not %ldc -conf=%S/inputs/section_aaa.conf %s 2>&1 | FileCheck %s --check-prefix=NO_SEC
// NO_SEC: Could not look up section '{{.*}}' nor the 'default' section in {{.*}}section_aaa.conf

// RUN: %ldc -conf=%S/inputs/override_default.conf -mtriple=x86-apple-windows-msvc -c -o- %s | FileCheck %s --check-prefix=OVERRIDE_DEFAULT
// OVERRIDE_DEFAULT: LDC - the LLVM D compiler


void foo()
{
}
