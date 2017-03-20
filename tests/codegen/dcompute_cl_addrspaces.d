// Test addrspace

// REQUIRES: target_SPIRV
// RUN: %ldc -c -mdcompute-targets=ocl-220 -m64 -output-ll -output-o %s && FileCheck %s --check-prefix=LL < kernels_ocl220_64.ll && llvm-spirv -to-text kernels_ocl220_64.spv && FileCheck %s --check-prefix=SPT < kernels_ocl220_64.spt
@compute(CompileFor.deviceOnly) module dcompute_cl_addrspaces;
import ldc.attributes;
import ldc.dcomputetypes;

// LL: %"ldc.dcomputetypes.Pointer!(0u, float).Pointer" = type { float* }
// LL: %"ldc.dcomputetypes.Pointer!(1u, float).Pointer" = type { float addrspace(1)* }
// LL: %"ldc.dcomputetypes.Pointer!(2u, float).Pointer" = type { float addrspace(2)* }
// LL: %"ldc.dcomputetypes.Pointer!(3u, immutable(float)).Pointer" = type { float addrspace(3)* }
// LL: %"ldc.dcomputetypes.Pointer!(4u, float).Pointer" = type { float addrspace(4)* }
// SPT: 3 TypeFloat [[FLOAT_ID:[0-9]+]] 32
// SPT: 2 TypeVoid [[VOID_ID:[0-9]+]]

//See section 3.7 of the SPIR-V Specification for the numbers in the 4th column.
// SPT: 4 TypePointer [[SHARED_FLOAT_POINTER_ID:[0-9]+]] 4 [[FLOAT_ID]]
// SPT: 4 TypePointer [[CONSTANT_FLOAT_POINTER_ID:[0-9]+]] 0 [[FLOAT_ID]]
// SPT: 4 TypePointer [[GLOABL_FLOAT_POINTER_ID:[0-9]+]] 5 [[FLOAT_ID]]
// SPT: 4 TypePointer [[GENERIC_FLOAT_POINTER_ID:[0-9]+]] 8 [[FLOAT_ID]]
// SPT: 4 TypePointer [[PRIVATE_FLOAT_POINTER_ID:[0-9]+]] 7 [[FLOAT_ID]]

//PrivatePointer and friends are { T addrspace(n)* }
// SPT: 3 TypeStruct [[STRUCT_PRIVATE_FLOAT_POINTER_ID:[0-9]+]] [[PRIVATE_FLOAT_POINTER_ID]]
// SPT: 3 TypeStruct [[STRUCT_SHARED_FLOAT_POINTER_ID:[0-9]+]] [[SHARED_FLOAT_POINTER_ID]]
// SPT: 3 TypeStruct [[STRUCT_CONSTANT_FLOAT_POINTER_ID:[0-9]+]] [[CONSTANT_FLOAT_POINTER_ID]]
// SPT: 3 TypeStruct [[STRUCT_GLOBAL_FLOAT_POINTER_ID:[0-9]+]] [[GLOBAL_FLOAT_POINTER_ID]]
// SPT: 3 TypeStruct [[STRUCT_GENERIC_FLOAT_POINTER_ID:[0-9]+]] [[GENERIC_FLOAT_POINTER_ID]]

//void function({ T addrspace(n)* })
// SPT: 4 TypeFunction [[FOO_PRIVATE:[0-9]+]] [[VOID_ID]] [[STRUCT_PRIVATE_FLOAT_POINTER_ID]]
// SPT: 4 TypeFunction [[FOO_GLOBAL:[0-9]+]] [[VOID_ID]] [[STRUCT_GLOBAL_FLOAT_POINTER_ID]]
// SPT: 4 TypeFunction [[FOO_SHARED:[0-9]+]] [[VOID_ID]] [[STRUCT_SHARED_FLOAT_POINTER_ID]]
// SPT: 4 TypeFunction [[FOO_CONSTANT:[0-9]+]] [[VOID_ID]] [[STRUCT_CONSTANT_FLOAT_POINTER_ID]]
// SPT: 4 TypeFunction [[FOO_GENERIC:[0-9]+]] [[VOID_ID]] [[STRUCT_GENERIC_FLOAT_POINTER_ID]]

void foo(PrivatePointer!float f) {
    // LL: load float, float*
    // SPT: 5 Function [[VOID_ID]] 0 [0-9]+ [[FOO_PRIVATE]]
    float g = *f;
}
void foo(GlobalPointer!float f) {
    // LL: load float, float addrspace(1)*
    // SPT: 5 Function [[VOID_ID]] 0 [0-9]+ [[FOO_GLOBAL]]
    float g = *f;
}
void foo(SharedPointer!float f) {
    // LL: load float, float addrspace(2)*
    // SPT: 5 Function [[VOID_ID]] 0 [0-9]+ [[FOO_SHARED]]
    float g = *f;
}
void foo(ConstantPointer!float f) {
    // LL: load float, float addrspace(3)*
    // SPT: 5 Function [[VOID_ID]] 0 [0-9]+ [[FOO_CONSTANT]]
    float g = *f;
}
void foo(GenericPointer!float f) {
    // LL: load float, float addrspace(4)*
    // SPT: 5 Function [[VOID_ID]] 0 [0-9]+ [[FOO_GENERIC]]
    float g = *f;
}
