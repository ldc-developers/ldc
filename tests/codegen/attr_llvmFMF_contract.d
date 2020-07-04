// Tests the @ldc.attributes.llvmFastMathFlag("contract") UDA
// Also tests that adding this attribute indeed leads to LLVM optimizing it to a fused multiply-add for a simple case.

// REQUIRES: target_X86

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s --check-prefix LLVM < %t.ll
// RUN: %ldc -betterC -mtriple=x86_64-linux-gnu -mattr=+fma -O3 -release -c -output-s -of=%t.s %s && FileCheck %s --check-prefix ASM < %t.s

import ldc.attributes;

// LLVM-LABEL: define{{.*}} @{{.*}}contract
// ASM-LABEL: contract:
@llvmFastMathFlag("contract")
extern(C) double contract(double a, double b, double c)
{
// LLVM: fmul contract double
// LLVM: fadd contract double
// ASM: vfmadd
    return a * b + c;
}
