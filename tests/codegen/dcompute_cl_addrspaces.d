// See GH issue #2709

// FIXME: hits an assertion for LLVM 18: https://github.com/llvm/llvm-project/issues/87315
// UNSUPPORTED: atleast_llvm1800 && atmost_llvm1809

// REQUIRES: target_SPIRV && atleast_llvm1600
// RUN: %ldc -c -mdcompute-targets=ocl-220 -m64 -mdcompute-file-prefix=addrspace_new -output-ll -output-o %s && FileCheck %s --check-prefix=LL < addrspace_new_ocl220_64.ll \
// RUN: && %llc addrspace_new_ocl220_64.ll -mtriple=spirv64-unknown-unknown -O0 -o - | FileCheck %s --check-prefix=SPT
@compute(CompileFor.deviceOnly) module dcompute_cl_addrspaces;
import ldc.dcompute;

// LL: %"ldc.dcompute.Pointer!(AddrSpace.Private, float).Pointer" = type { ptr }
// LL: %"ldc.dcompute.Pointer!(AddrSpace.Global, float).Pointer" = type { ptr addrspace(1) }
// LL: %"ldc.dcompute.Pointer!(AddrSpace.Shared, float).Pointer" = type { ptr addrspace(2) }
// LL: %"ldc.dcompute.Pointer!(AddrSpace.Constant, immutable(float)).Pointer" = type { ptr addrspace(3) }
// LL: %"ldc.dcompute.Pointer!(AddrSpace.Generic, float).Pointer" = type { ptr addrspace(4) }

// SPT-DAG: %{{[0-9]+}} = OpTypeVoid
// SPT-DAG: %{{[0-9]+}} = OpTypeFloat 32

//See section 3.7 of the SPIR-V Specification for the numbers in the 4th column.
// SPT-DAG: %{{[0-9]+}} = OpTypePointer CrossWorkgroup %{{[0-9]+}}
// SPT-DAG: %{{[0-9]+}} = OpTypePointer UniformConstant %{{[0-9]+}}
// SPT-DAG: %{{[0-9]+}} = OpTypePointer Workgroup %{{[0-9]+}}
// SPT-DAG: %{{[0-9]+}} = OpTypePointer Generic %{{[0-9]+}}

//void function({ T addrspace(n)* })

void foo(PrivatePointer!float f) {
    // LL: load float, ptr
    float g = *f;
}

void foo(GlobalPointer!float f) {
    // LL: load float, ptr addrspace(1)
    float g = *f;
}

void foo(SharedPointer!float f) {
    // LL: load float, ptr addrspace(2)
    float g = *f;
}

void foo(ConstantPointer!float f) {
    // LL: load float, ptr addrspace(3)
    float g = *f;
}

void foo(GenericPointer!float f) {
    // LL: load float, ptr addrspace(4)
    float g = *f;
}
