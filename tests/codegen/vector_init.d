// Make sure vector initializer llvm::Constants are generated correctly (GitHub #2101).
// RUN: %ldc -c -O3 -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

alias D2 = __vector(double[2]);

// CHECK: @_D11vector_init12ImplicitInit6__initZ =
// double.init (as well as float.init) may be silenced from a signalling to a quiet NaN,
// so don't check the most significant mantissa bit (quietness), but make sure the
// second-most significant one is (still) set.
// See https://github.com/ldc-developers/ldc/pull/2207.
// CHECK-SAME: { <2 x double> <double 0x7FF{{4|C}}000000000000, double 0x7FF{{4|C}}000000000000> }
struct ImplicitInit { D2 a; }

// CHECK: @_D11vector_init12ExplicitInit6__initZ =
// CHECK-SAME: { <2 x double> <double 0x7FF{{4|C}}000000000000, double 0x7FF{{4|C}}000000000000> }
struct ExplicitInit { D2 a = D2.init; }

// CHECK: @_D11vector_init10SplatValue6__initZ =
// CHECK-SAME: { <2 x double> <double 1.000000e+00, double 1.000000e+00> }
struct SplatValue { D2 a = 1.0; }

// CHECK: @_D11vector_init13ElementValues6__initZ =
// CHECK-SAME: { <2 x double> <double 1.000000e+00, double 2.000000e+00> }
struct ElementValues { D2 a = [1.0, 2.0]; }
