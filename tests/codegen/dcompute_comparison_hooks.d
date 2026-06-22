// Array `==` / `!=` / `<` / `>` / `<=` / `>=` in @compute device code lower to the
// druntime hooks core.internal.array.equality.__equals and
// core.internal.array.comparison.__cmp (plus helpers like isEqual/at). Those hooks
// live in host-only druntime modules but are device-legal. Regression guard: they
// used to be either rejected during dcompute semantic analysis or emitted as hollow
// `declare`-only stubs. This test asserts the device IR contains real `define`d
// bodies (never bare `declare`s) for every element type that actually instantiates
// the templates.
//
// float / double / real / float[N] / struct-with-opEquals / struct-with-float-field
// all bypass the int[] memcmp fast path, so the genuine __equals / __cmp templates
// are instantiated and codegen'd for the device. (Plain POD structs and integral
// element types take the inline-memcmp path instead -- see compilable/dcompute_comparison_hooks.d.)
//
// REQUIRES: target_NVPTX
// RUN: %ldc -mdcompute-targets=cuda-700 -m64 -output-ll -output-o -c \
// RUN:   -mdcompute-file-prefix=comparison_hooks -Iinputs %s
// RUN: FileCheck %s < comparison_hooks_cuda700_64.ll

@compute(CompileFor.deviceOnly) module dcompute_comparison_hooks;
import ldc.dcompute;

struct SEq    { int x; bool opEquals(ref const SEq o) const { return x == o.x; } }
struct SFloat { float f; }
struct SCmp   { int x; int opCmp(ref const SCmp o) const { return x < o.x ? -1 : (x > o.x ? 1 : 0); } }

// ---- __equals (== / !=) ----------------------------------------------------

// Dynamic float[] equality -> __equals!(float,float).
// CHECK-DAG: define{{.*}}ptx_device{{.*}}@_D4core8internal5array8equality{{.*}}__equalsTfTf
@kernel void eq_float(float[] a, float[] b, bool* o)    { *o = (a == b); }

// Dynamic double[] equality -> __equals!(double,double).
// CHECK-DAG: define{{.*}}ptx_device{{.*}}@_D4core8internal5array8equality{{.*}}__equalsTdTd
@kernel void eq_double(double[] a, double[] b, bool* o) { *o = (a == b); }

// Static float[4] equality -> __equals!(float,float, len, len).
// CHECK-DAG: define{{.*}}ptx_device{{.*}}@_D4core8internal5array8equality{{.*}}__equalsTfTfVmi4
@kernel void eq_static(float[4] a, float[4] b, bool* o) { *o = (a == b); }

// Struct with custom opEquals -> __equals!(SEq,SEq), element-wise via the user opEquals.
// CHECK-DAG: define{{.*}}ptx_device{{.*}}@_D4core8internal5array8equality{{.*}}__equalsTS{{.*}}3SEq
@kernel void eq_opequals(SEq[] a, SEq[] b, bool* o)     { *o = (a == b); }

// Struct with a float field (forces element-wise compare, not memcmp) -> __equals!(SFloat,SFloat).
// CHECK-DAG: define{{.*}}ptx_device{{.*}}@_D4core8internal5array8equality{{.*}}__equalsTS{{.*}}6SFloat
@kernel void eq_floatfield(SFloat[] a, SFloat[] b, bool* o) { *o = (a == b); }

// `!=` lowers to the same __equals!(float,float) hook (negated). No NEW symbol needed
// (shares eq_float's instantiation), but it must compile and stay defined.
@kernel void ne_float(float[] a, float[] b, bool* o)    { *o = (a != b); }

// The element comparator helper isEqual must also be defined (reached transitively
// from __equals, lives in the same host-only equality module).
// CHECK-DAG: define{{.*}}ptx_device{{.*}}@_D4core8internal5array8equality{{.*}}isEqual

// ---- __cmp (< / > / <= / >=) -----------------------------------------------

// float[] ordering -> __cmp!float. All four relational operators lower to the
// same __cmp!float hook, so one define covers <, >, <=, >=.
// CHECK-DAG: define{{.*}}ptx_device{{.*}}@_D4core8internal5array10comparison{{.*}}__cmpTf
@kernel void cmp_lt(float[] a, float[] b, bool* o) { *o = (a <  b); }
@kernel void cmp_gt(float[] a, float[] b, bool* o) { *o = (a >  b); }
@kernel void cmp_le(float[] a, float[] b, bool* o) { *o = (a <= b); }
@kernel void cmp_ge(float[] a, float[] b, bool* o) { *o = (a >= b); }

// double[] ordering -> __cmp!double.
// CHECK-DAG: define{{.*}}ptx_device{{.*}}@_D4core8internal5array10comparison{{.*}}__cmpTd
@kernel void cmp_double(double[] a, double[] b, bool* o) { *o = (a < b); }

// int[] ordering -> __cmp!int (ordering goes through __cmp, NOT the memcmp fast path
// that int[] `==` uses).
// CHECK-DAG: define{{.*}}ptx_device{{.*}}@_D4core8internal5array10comparison{{.*}}__cmpTi
@kernel void cmp_int(int[] a, int[] b, bool* o) { *o = (a < b); }

// Struct with custom opCmp -> __cmp!(SCmp), element-wise via the user opCmp.
// CHECK-DAG: define{{.*}}ptx_device{{.*}}@_D4core8internal5array10comparison{{.*}}__cmpTS{{.*}}4SCmp
@kernel void cmp_opcmp(SCmp[] a, SCmp[] b, bool* o) { *o = (a < b); }

// None of the hooks may be left as hollow `declare`-only stubs.
// CHECK-NOT: declare{{.*}}__equals
// CHECK-NOT: declare{{.*}}__cmp
