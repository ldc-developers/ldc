// Tests inline IR + math optimizations

// REQUIRES: target_X86

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s --check-prefix LLVM < %t.ll
// RUN: %ldc -mtriple=x86_64-linux-gnu -mattr=+fma -O3 -release -c -output-s -of=%t.s %s && FileCheck %s --check-prefix ASM < %t.s

import ldc.attributes;
import ldc.llvmasm;

// Test that internal @inline.ir.*" functions for the inlined IR pieces are always inlined and are not present as a global symbol
// LLVM-NOT: @inline.ir.
// LLVM-NOT: alwaysinline


// __ir should inherit the enclosing function attributes, thus preserving the enclosing function attributes after inlining.
// LLVM-LABEL: define{{.*}} @dot
// LLVM-SAME: #[[UNSAFEFPMATH:[0-9]+]]
// ASM-LABEL: dot:
@llvmAttr("unsafe-fp-math", "true")
extern (C) double dot(double[] a, double[] b)
{
    double s = 0;

// ASM: vfmadd{{[123][123][123]}}pd
    foreach (size_t i; 0 .. a.length)
    {
        s = __ir!(`%p = fmul fast double %0, %1
                   %r = fadd fast double %p, %2
                   ret double %r`, double)(a[i], b[i], s);
    }
    return s;
}

// LLVM-LABEL: define{{.*}} @features
// LLVM-SAME: #[[FEAT:[0-9]+]]
@target("fma")
extern (C) double features(double[] a, double[] b)
{
    double s = 0;
    foreach (size_t i; 0 .. a.length)
    {
        s = __ir!(`%p = fmul fast double %0, %1
                   %r = fadd fast double %p, %2
                   ret double %r`, double)(a[i], b[i], s);
    }
    return s;
}

// Test that inline IR works when calling function has special attributes defined for its parameters
// LLVM-LABEL: define{{.*}} @dot160
// ASM-LABEL: dot160:
extern (C) double dot160(double[160] a, double[160] b)
{
    double s = 0;

// ASM-NOT: vfmadd
    foreach (size_t i; 0 .. a.length)
    {
        s = __ir!(`%p = fmul double %0, %1
                   %r = fadd double %p, %2
                   ret double %r`, double)(a[i], b[i], s);
    }
    return s;
}

// Test inline IR alias defined outside any function
alias __ir!(`%p = fmul fast double %0, %1
             %r = fadd fast double %p, %2
             ret double %r`,
             double, double, double, double) muladdFast;
alias __ir!(`%p = fmul double %0, %1
             %r = fadd double %p, %2
             ret double %r`,
             double, double, double, double) muladd;

// LLVM-LABEL: define{{.*}} @aliasInlineUnsafe
// LLVM-SAME: #[[UNSAFEFPMATH]]
// ASM-LABEL: aliasInlineUnsafe:
@llvmAttr("unsafe-fp-math", "true")
extern (C) double aliasInlineUnsafe(double[] a, double[] b)
{
    double s = 0;

// ASM: vfmadd{{[123][123][123]}}pd
    foreach (size_t i; 0 .. a.length)
    {
        s = muladdFast(a[i], b[i], s);
    }
    return s;
}

// LLVM-LABEL: define{{.*}} @aliasInlineSafe
// LLVM-SAME: #[[NO_UNSAFEFPMATH:[0-9]+]]
// ASM-LABEL: aliasInlineSafe:
extern (C) double aliasInlineSafe(double[] a, double[] b)
{
    double s = 0;

// ASM-NOT: vfmadd{{[123][123][123]}}pd
    foreach (size_t i; 0 .. a.length)
    {
        s = muladd(a[i], b[i], s);
    }
    return s;
}

// Make sure an enclosing function's 'noinline' attribute isn't copied to
// the inlined IR function (having 'alwaysinline') (issue #1711).
double neverInlinedEnclosingFunction()
{
    pragma(inline, false);
    return muladd(1.0, 2.0, 3.0);
}

// LLVM: attributes #[[UNSAFEFPMATH]] ={{.*}} "unsafe-fp-math"="true"
// LLVM: attributes #[[FEAT]] ={{.*}} "target-features"="{{.*}}+fma{{.*}}"

// LLVM: attributes #[[NO_UNSAFEFPMATH]] =
// LLVM-NOT: "unsafe-fp-math"="true"
