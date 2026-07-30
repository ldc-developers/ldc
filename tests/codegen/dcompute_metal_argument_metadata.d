// REQUIRES: target_AArch64
// REQUIRES: llvm20

// RUN: %ldc -c -mdcompute-targets=metal-400 --mdcompute-file-prefix=argument_metadata -output-ll -of=%t.ll %s
// RUN: FileCheck %s --check-prefix=AIR < argument_metadata_metal400_64.air

@compute(CompileFor.deviceOnly) module kernels;

import ldc.dcompute;

// AIR-LABEL: define {{.*}} @{{.*}}test_multi_arg_kernel{{.*}}(
@kernel()
void test_multi_arg_kernel(GlobalPointer!float data, SharedPointer!float shared_data, float scalar_value)
{
    data[0] = scalar_value;
    shared_data[0] = scalar_value;
}

// AIR-DAG: !air.kernel = !{[[KERNEL:![0-9]+]]}

// AIR-DAG: [[KERNEL]] = !{ptr @{{.*}}test_multi_arg_kernel{{.*}}, [[EMPTY:![0-9]+]], [[ARGS_ROOT:![0-9]+]]}

// AIR-DAG: [[ARGS_ROOT]] = !{[[ARG_0:![0-9]+]], [[ARG_1:![0-9]+]], [[ARG_2:![0-9]+]]}

// COM: expects address_space: 1 for global pointer
// AIR-DAG: [[ARG_0]] = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"data"}

// COM: expects address_space: 3 for shared pointer
// AIR-DAG: [[ARG_1]] = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"shared_data"}

// COM: expects address_space: 2 for scalar values
// AIR-DAG: [[ARG_2]] = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"scalar_value"}

