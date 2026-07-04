// hostAndDevice companion for compilable/dcompute_comparison_hooks.d.
// The array comparison/equality hooks (__equals/__cmp) must also compile when the
// @compute module targets both host and device, not just deviceOnly.
@compute(CompileFor.hostAndDevice) module inputs.comparison_hooks_had;
import ldc.dcompute;

struct SEq { int x; bool opEquals(ref const SEq o) const { return x == o.x; } }

@kernel void had_eq    (float[] a, float[] b, bool* o) { *o = (a == b); }
@kernel void had_cmp   (float[] a, float[] b, bool* o) { *o = (a <  b); }
@kernel void had_struct(SEq[]   a, SEq[]   b, bool* o) { *o = (a == b); }
