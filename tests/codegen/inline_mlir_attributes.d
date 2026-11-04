// Tests that the generated `@inline.mlir.N` helpers inherit the function
// attributes of the enclosing function. Some attributes (like
// "unsafe-fp-math") are AND-merged into the caller when inlining, so without
// copying them the enclosing function would lose them.
// Also tests that MLIR-level instruction flags (fastmath) reach LLVM IR.

// REQUIRES: MLIR

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s --implicit-check-not=@inline.mlir < %t.ll

import ldc.attributes;
import ldc.llvmasm;

alias muladdFast = __mlir!(`
    %p = arith.mulf %arg0, %arg1 fastmath<fast> : f64
    %r = arith.addf %p, %arg2 fastmath<fast> : f64
    return %r : f64
`, double, double, double, double);

alias muladd = __mlir!(`
    %p = arith.mulf %arg0, %arg1 : f64
    %r = arith.addf %p, %arg2 : f64
    return %r : f64
`, double, double, double, double);

// CHECK-LABEL: define{{.*}} double @unsafe(
// CHECK-SAME: #[[UNSAFE:[0-9]+]]
@llvmAttr("unsafe-fp-math", "true")
extern (C) double unsafe(double a, double b, double c)
{
    // CHECK: fmul fast double
    // CHECK: fadd fast double
    return muladdFast(a, b, c);
}

// CHECK-LABEL: define{{.*}} double @safe(
// CHECK-SAME: #[[SAFE:[0-9]+]]
extern (C) double safe(double a, double b, double c)
{
    // CHECK: fmul double
    // CHECK: fadd double
    return muladd(a, b, c);
}

// The plain (non-fastmath) MLIR snippet inlined into an unsafe-fp-math function
// must not drop the caller's attribute.
// CHECK-LABEL: define{{.*}} double @unsafe_plain(
// CHECK-SAME: #[[UNSAFE]]
@llvmAttr("unsafe-fp-math", "true")
extern (C) double unsafe_plain(double a, double b, double c)
{
    // CHECK: fmul double
    // CHECK: fadd double
    return muladd(a, b, c);
}

// CHECK: attributes #[[UNSAFE]] ={{.*}} "unsafe-fp-math"="true"
// CHECK: attributes #[[SAFE]] =
