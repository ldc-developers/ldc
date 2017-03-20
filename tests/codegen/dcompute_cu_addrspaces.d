// Test addrspace

// REQUIRES: target_NVPTX
// RUN: %ldc -c -mdcompute-targets=cuda-350 -m64 -output-ll -output-o %s && FileCheck %s --check-prefix=LL < kernels_cuda350_64.ll && FileCheck %s --check-prefix=PTX < kernels_cuda350_64.ptx
@compute(CompileFor.deviceOnly) module dcompute_cu_addrspaces;
import ldc.attributes;
import ldc.dcomputetypes;

// LL: %"ldc.dcomputetypes.Pointer!(0u, float).Pointer" = type { float addrspace(5)* }
// LL: %"ldc.dcomputetypes.Pointer!(1u, float).Pointer" = type { float addrspace(1)* }
// LL: %"ldc.dcomputetypes.Pointer!(2u, float).Pointer" = type { float addrspace(3)* }
// LL: %"ldc.dcomputetypes.Pointer!(3u, immutable(float)).Pointer" = type { float addrspace(4)* }
// LL: %"ldc.dcomputetypes.Pointer!(4u, float).Pointer" = type { float* }

void foo(PrivatePointer!float f) {
    // LL: load float, float addrspace(5)*
    // PTX: ld.local.f32
    float g = *f;
}
void foo(GlobalPointer!float f) {
    // LL: load float, float addrspace(1)*
    // PTX: ld.global.f32
    float g = *f;
}
void foo(SharedPointer!float f) {
    // LL: load float, float addrspace(3)*
    // PTX: ld.shared.f32
    float g = *f;
}
void foo(ConstantPointer!float f) {
    // LL: load float, float addrspace(4)*
    // PTX: ld.const.f32
    float g = *f;
}
void foo(GenericPointer!float f) {
    // LL: load float, float*
    // PTX: ld.f32
    float g = *f;
}
