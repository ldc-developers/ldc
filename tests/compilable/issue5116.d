// Without this patch, template instantiations from non-@compute modules (e.g.
// __equals from core.internal.array.equality) were walked, producing spurious errors.
// And indirect calls via function pointers inside those bodies additionally caused a
// null-pointer dereference crash.

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
