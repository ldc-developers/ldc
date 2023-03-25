// Very basic test to check that instrinsics include file is made correctly.
// Related to https://github.com/ldc-developers/ldc/issues/4347

// Just do SemA, no codegen, such that it works on all CI systems.
// RUN: %ldc -o- %s

import core.simd;
static import ldc.gccbuiltins_aarch64;
static import ldc.gccbuiltins_arm;
static import ldc.gccbuiltins_mips;
static import ldc.gccbuiltins_nvvm;
static import ldc.gccbuiltins_ppc;
static import ldc.gccbuiltins_x86;

short2 s2;
short8 s8;
double2 d2;
void* ptr;

void main()
{
    ldc.gccbuiltins_aarch64.__builtin_arm_isb(1);

    ldc.gccbuiltins_arm.__builtin_arm_dmb(2);

    short2 mips = ldc.gccbuiltins_mips.__builtin_mips_addq_s_ph(s2, s2);

    double nvvm = ldc.gccbuiltins_nvvm.__nvvm_fma_rz_d(1.0, 2.0, 3.0);

    short8 ppc8 = ldc.gccbuiltins_ppc.__builtin_altivec_crypto_vpmsumh(s8, s8);

    ldc.gccbuiltins_x86.__builtin_ia32_lfence();
    ldc.gccbuiltins_x86.__builtin_ia32_umonitor(ptr);
    double2 x86 = ldc.gccbuiltins_x86.__builtin_ia32_maxpd(d2, d2);
}
