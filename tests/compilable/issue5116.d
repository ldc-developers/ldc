// Regression test for issue #5116: defining a struct with a static array
// field in a @compute module previously caused a spurious semantic error
// ("can only call functions from other `@compute` modules") followed by a
// null-pointer dereference crash in DComputeSemanticAnalyser::visit(CallExp*).
// The crash happened because compiler-generated support functions (__xopEquals)
// triggered instantiation of __equals templates from core.internal.array.equality,
// whose body contains an indirect call through a function pointer (e->f == null).

// REQUIRES: target_NVPTX
// RUN: %ldc -mdcompute-targets=cuda-350 %s

@compute(CompileFor.deviceOnly) module tests.compilable.issue5116;
import ldc.dcompute;

private enum N = 16u;

struct S {
    float[N] data;
}
