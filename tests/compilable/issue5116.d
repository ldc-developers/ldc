// Issue #5116: defining a struct with a static array field in a @compute
// module caused DMD to generate __xopEquals, dragging in template
// instantiations from core.internal.array.equality. The semantic walker
// then either reported spurious errors on the nested __equals calls or
// crashed on a null function pointer inside the instantiated body.

// REQUIRES: target_NVPTX
// RUN: %ldc -mdcompute-targets=cuda-350 %s

@compute(CompileFor.deviceOnly) module tests.compilable.issue5116;
import ldc.dcompute;

private enum N = 16u;

struct S {
    float[N] data;
}

@kernel void testEqualExp() {
    float[N] a, b;
    bool c = (a == b);
}

@kernel void testExplicitEquals() {
    float[N] a, b;
    bool c = __equals(a, b);
}
