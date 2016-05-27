// Test vectorized fused multiply-add in a simple dot product routine

// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64-linux-gnu -mattr=+fma -O3 -release -c -output-s -of=%t.s %s && FileCheck %s --check-prefix ASM < %t.s

import ldc.attributes;

// ASM-LABEL: dot:
@fastmath
extern (C) double dot(double[] a, double[] b)
{
    double s = 0;
// ASM: vfmadd{{[123][123][123]}}pd
    foreach (size_t i; 0 .. a.length)
    {
        s += a[i] * b[i];
    }
    return s;
// ASM: ret
}
