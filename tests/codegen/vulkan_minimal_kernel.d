// REQUIRES: target_SPIRV
// RUN: %ldc -c -m64 -mdcompute-targets=vulkan-130 -mdcompute-file-prefix=vulkan_out -output-o %s
// RUN: spirv-dis vulkan_out_vulkan130_64.spv -o - | FileCheck %s

@compute(CompileFor.deviceOnly) module vulkan_minimal_kernel;
import ldc.dcompute;

// CHECK: OpEntryPoint GLCompute %[[WRAPPER:[a-zA-Z0-9_]+]] "{{.*}}minimal_kernel{{.*}}"

// CHECK: %[[CORE:[a-zA-Z0-9_]+]] = OpFunction %void
// CHECK: OpStore
// CHECK: OpReturn
// CHECK: OpFunctionEnd

// CHECK: %[[WRAPPER]] = OpFunction %void
// CHECK: OpFunctionCall %void %[[CORE]]
// CHECK: OpReturn
// CHECK: OpFunctionEnd

@kernel() void minimal_kernel(GlobalPointer!float output) {
    *output = 42.0f;
}
