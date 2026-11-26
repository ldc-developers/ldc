// REQUIRES: target_SPIRV && atleast_llvm23
// RUN: %ldc -c -m64 -mdcompute-targets=vulkan-130 -mdcompute-file-prefix=vulkan_va_out -output-o %s
// RUN: spirv-dis vulkan_va_out_vulkan130_64.spv -o - | FileCheck %s

@compute(CompileFor.deviceOnly) module vulkan_vector_add;
import ldc.dcompute;

// CHECK: OpEntryPoint GLCompute %[[WRAPPER:[a-zA-Z0-9_]+]] "{{.*}}vector_add{{.*}}"

@kernel() void vector_add(
    GlobalPointer!float A0, GlobalPointer!float A1, GlobalPointer!float A2,
    GlobalPointer!float B0, GlobalPointer!float B1, GlobalPointer!float B2,
    GlobalPointer!float C0, GlobalPointer!float C1, GlobalPointer!float C2
) {
    *C0 = *A0 + *B0;
    *C1 = *A1 + *B1;
    *C2 = *A2 + *B2;
}
