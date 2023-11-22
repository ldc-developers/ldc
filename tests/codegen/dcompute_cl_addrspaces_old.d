// See GH issue #2709

// REQUIRES: target_SPIRV && atmost_llvm1509
// RUN: %ldc -c -mdcompute-targets=ocl-220 -m64 -mdcompute-file-prefix=addrspace_old -output-ll -output-o %s && FileCheck %s --check-prefix=LL < addrspace_ocl220_64.ll \
// RUN: && %llvm-spirv -to-text addrspace_old_ocl220_64.spv && FileCheck %s --check-prefix=SPT < addrspace_ocl220_64.spt
@compute(CompileFor.deviceOnly) module dcompute_cl_addrspaces_old;
import ldc.dcompute;

// LL: %"ldc.dcompute.Pointer!(AddrSpace.Private, float).Pointer" = type { float* }
// LL: %"ldc.dcompute.Pointer!(AddrSpace.Global, float).Pointer" = type { float addrspace(1)* }
// LL: %"ldc.dcompute.Pointer!(AddrSpace.Shared, float).Pointer" = type { float addrspace(2)* }
// LL: %"ldc.dcompute.Pointer!(AddrSpace.Constant, immutable(float)).Pointer" = type { float addrspace(3)* }
// LL: %"ldc.dcompute.Pointer!(AddrSpace.Generic, float).Pointer" = type { float addrspace(4)* }

// SPT-DAG: 2 TypeVoid [[VOID_ID:[0-9]+]]
// SPT-DAG: 3 TypeFloat [[FLOAT_ID:[0-9]+]] 32

//See section 3.7 of the SPIR-V Specification for the numbers in the 4th column.
// SPT-DAG: 4 TypePointer [[SHARED_FLOAT_POINTER_ID:[0-9]+]] 4 [[FLOAT_ID]]
// SPT-DAG: 4 TypePointer [[CONSTANT_FLOAT_POINTER_ID:[0-9]+]] 0 [[FLOAT_ID]]
// SPT-DAG: 4 TypePointer [[GLOBAL_FLOAT_POINTER_ID:[0-9]+]] 5 [[FLOAT_ID]]
// SPT-DAG: 4 TypePointer [[GENERIC_FLOAT_POINTER_ID:[0-9]+]] 8 [[FLOAT_ID]]
// SPT-DAG: 4 TypePointer [[PRIVATE_FLOAT_POINTER_ID:[0-9]+]] 7 [[FLOAT_ID]]

//void function({ T addrspace(n)* })
// SPT-DAG: 4 TypeFunction [[FOO_PRIVATE:[0-9]+]] [[VOID_ID]] [[PRIVATE_FLOAT_POINTER_ID]]
// SPT-DAG: 4 TypeFunction [[FOO_GLOBAL:[0-9]+]] [[VOID_ID]] [[GLOBAL_FLOAT_POINTER_ID]]
// SPT-DAG: 4 TypeFunction [[FOO_SHARED:[0-9]+]] [[VOID_ID]] [[SHARED_FLOAT_POINTER_ID]]
// SPT-DAG: 4 TypeFunction [[FOO_CONSTANT:[0-9]+]] [[VOID_ID]] [[CONSTANT_FLOAT_POINTER_ID]]
// SPT-DAG: 4 TypeFunction [[FOO_GENERIC:[0-9]+]] [[VOID_ID]] [[GENERIC_FLOAT_POINTER_ID]]

void foo(PrivatePointer!float f) {
    // LL: load float, float*
    // SPT-DAG: 5 Function [[VOID_ID]] {{[0-9]+}} 0 [[FOO_PRIVATE]]
    float g = *f;
}

void foo(GlobalPointer!float f) {
    // LL: load float, float addrspace(1)*
    // SPT-DAG: 5 Function [[VOID_ID]] {{[0-9]+}} 0 [[FOO_GLOBAL]]
    float g = *f;
}

void foo(SharedPointer!float f) {
    // LL: load float, float addrspace(2)*
    // SPT-DAG: 5 Function [[VOID_ID]] {{[0-9]+}} 0 [[FOO_SHARED]]
    float g = *f;
}

void foo(ConstantPointer!float f) {
    // LL: load float, float addrspace(3)*
    // SPT-DAG: 5 Function [[VOID_ID]] {{[0-9]+}} 0 [[FOO_CONSTANT]]
    float g = *f;
}

void foo(GenericPointer!float f) {
    // LL: load float, float addrspace(4)*
    // SPT-DAG: 5 Function [[VOID_ID]] {{[0-9]+}} 0 [[FOO_GENERIC]]
    float g = *f;
}
