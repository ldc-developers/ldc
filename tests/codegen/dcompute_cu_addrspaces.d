// Test addrspace

// REQUIRES: target_NVPTX
// RUN: %ldc -mdcompute-targets=cuda350 -m64 -output-ll -output-o%s && FileCheck %s --check-prefix=LL < kernels_cuda350_64.ll && FileCheck %s --check-prefix=PTX < kernels_cuda350_64.ptx
@compute(CompileFor.deviceOnly) module dcompute_cu_addrspaces;
import ldc.attributes;
import ldc.dcomputetypes;

// LL: %"ldc.dcomputetypes.Pointer!(1u, float).Pointer" = type { float* }
// PTX: foo
void foo(SharedPointer!float f) {
    *f = 0.0;
}
