// Companion to codegen/dcompute_comparison_hooks.d. That test proves the array
// comparison/equality hooks emit defined device bodies; this one proves the
// surrounding compile-ok matrix:
//   * the hooks compile in a deviceOnly @compute module, used directly in a
//     @kernel AND in a called non-kernel @compute helper,
//   * the hooks also compile in a hostAndDevice @compute module
//     (inputs/comparison_hooks_had.d),
//   * the Cat-2 "controls": integral element `==` takes the inline memcmp fast
//     path and never instantiates __equals at all.
//
// REQUIRES: target_NVPTX

// deviceOnly: hooks in a kernel and in a called @compute helper must compile.
// RUN: %ldc -c -mdcompute-targets=cuda-700 %s

// hostAndDevice: the same hooks must compile when the module targets both.
// RUN: %ldc -c -mdcompute-targets=cuda-700 -I%S %S/inputs/comparison_hooks_had.d

// Cat-2 control: int[]/int[4]/byte[] `==` lower to an inline memcmp; the __equals
// template is NEVER instantiated. Assert the device IR has no __equals symbol but
// does contain memcmp, documenting the bypass so future readers aren't surprised.
// RUN: %ldc -mdcompute-targets=cuda-700 -m64 -output-ll -output-o -c \
// RUN:   -mdcompute-file-prefix=cmphooks_ctrl -d-version=Controls %s
// RUN: FileCheck %s --check-prefix=CTRL < cmphooks_ctrl_cuda700_64.ll

@compute(CompileFor.deviceOnly) module dcompute_comparison_hooks;
import ldc.dcompute;

struct SEq { int x; bool opEquals(ref const SEq o) const { return x == o.x; } }

version (Controls)
{
    // CTRL-NOT: __equals
    // CTRL: memcmp
    @kernel void c_int (int[]  a, int[]  b, bool* o) { *o = (a == b); }
    @kernel void c_int4(int[4] a, int[4] b, bool* o) { *o = (a == b); }
    @kernel void c_byte(byte[] a, byte[] b, bool* o) { *o = (a == b); }
}
else
{
    // Hook used in a non-kernel @compute helper that the kernel calls.
    bool eqHelper(float[] a, float[] b) { return a == b; }
    int  cmpHelper(double[] a, double[] b) { return a < b ? -1 : 1; }

    // Hook used directly in a kernel, across element types.
    @kernel void k_float (float[]  a, float[]  b, bool* o) { *o = (a == b); }
    @kernel void k_double(double[] a, double[] b, bool* o) { *o = (a <  b); }
    @kernel void k_struct(SEq[]    a, SEq[]    b, bool* o) { *o = (a == b); }
    @kernel void k_helper(float[]  a, float[]  b, double[] c, double[] d, bool* o)
    {
        *o = eqHelper(a, b) && (cmpHelper(c, d) < 0);
    }
}
