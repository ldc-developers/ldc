// REQUIRES: target_X86

// RUN: %ldc -output-s -mtriple=x86_64-windows-msvc -of=%t_msvc.s %s
// RUN: FileCheck --check-prefix=COMMON --check-prefix=MSVC %s < %t_msvc.s
// RUN: %ldc -output-s -mtriple=x86_64-linux-gnu -of=%t_linux.s %s
// RUN: FileCheck --check-prefix=COMMON --check-prefix=LINUX %s < %t_linux.s

// COMMON: _D23dmd_inline_asm_fp_types3fooFfdeZv
void foo(float a, double b, real c)
{
    asm
    {
        // COMMON: flds
        fld a;
        // COMMON-NEXT: fldl
        fld b;
        // MSVC-NEXT: fldl
        // LINUX-NEXT: fldt
        fld c;
        ret;
    }
}

// COMMON: _D23dmd_inline_asm_fp_types3barFPvZv
void bar(void* ptr)
{
    asm
    {
        // COMMON: flds
        fld float ptr [ptr];
        // COMMON-NEXT: fldl
        fld double ptr [ptr];
        // MSVC-NEXT: fldl
        // LINUX-NEXT: fldt
        fld real ptr [ptr];
        // COMMON-NEXT: fldt
        fld extended ptr [ptr];
        ret;
    }
}
