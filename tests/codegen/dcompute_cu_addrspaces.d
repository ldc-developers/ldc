// REQUIRES: target_NVPTX
// RUN: %ldc -c -mdcompute-targets=cuda-350 -m64 -mdcompute-file-prefix=addrspace -output-ll -output-o %s && FileCheck %s --check-prefix=LL < addrspace_cuda350_64.ll && FileCheck %s --check-prefix=PTX < addrspace_cuda350_64.ptx
@compute(CompileFor.deviceOnly) module dcompute_cu_addrspaces;
import ldc.dcompute;

// LL: %"ldc.dcompute.Pointer!(AddrSpace.Private, float).Pointer" = type { ptr addrspace(5) }
// LL: %"ldc.dcompute.Pointer!(AddrSpace.Global, float).Pointer" = type { ptr addrspace(1) }
// LL: %"ldc.dcompute.Pointer!(AddrSpace.Shared, float).Pointer" = type { ptr addrspace(3) }
// LL: %"ldc.dcompute.Pointer!(AddrSpace.Constant, immutable(float)).Pointer" = type { ptr addrspace(4) }
// LL: %"ldc.dcompute.Pointer!(AddrSpace.Generic, float).Pointer" = type { ptr }

void foo(PrivatePointer!float f) {
    // LL: load float, ptr addrspace(5)
    // PTX: ld.local.f32
    float g = *f;
}
void foo(GlobalPointer!float f) {
    // LL: load float, ptr addrspace(1)
    // PTX: ld.global.f32
    float g = *f;
}
void foo(SharedPointer!float f) {
    // LL: load float, ptr addrspace(3)
    // PTX: ld.shared.f32
    float g = *f;
}
void foo(ConstantPointer!float f) {
    // LL: load float, ptr addrspace(4)
    // PTX: ld.const.f32
    float g = *f;
}
void foo(GenericPointer!float f) {
    // LL: load float, ptr
    // PTX: ld.f32
    float g = *f;
}
