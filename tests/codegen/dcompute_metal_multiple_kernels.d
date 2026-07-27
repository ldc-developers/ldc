// REQUIRES: target_AArch64
// REQUIRES: llvm20

// RUN: %ldc -c -mdcompute-targets=metal-400 -output-ll -of=%t.ll %s
// RUN: FileCheck %s --check-prefix=AIR < kernels_metal400_64.air

@compute(CompileFor.deviceOnly) module kernels;

import ldc.dcompute;

// AIR-LABEL: define {{.*}} @{{.*}}test_kernel_1{{.*}}(
@kernel()
void test_kernel_1(GlobalPointer!float data, float scalar_value)
{
    data[0] = scalar_value + 1;

    data[1] += scalar_value;
}


// AIR-LABEL: define {{.*}} @{{.*}}test_kernel_2{{.*}}(
@kernel()
void test_kernel_2(GlobalPointer!float data)
{
    data[0] = 0.0;
}

// AIR-DAG: !air.kernel = !{[[KERNEL_0:![0-9]+]], [[KERNEL_1:![0-9]+]]}

// AIR-DAG: [[KERNEL_0]] = !{ptr @{{.*}}test_kernel_1{{.*}}, [[EMPTY:![0-9]+]], [[ARGS_ROOT_OF_KERNEL_0:![0-9]+]]}

// AIR-DAG: [[KERNEL_1]] = !{ptr @{{.*}}test_kernel_2{{.*}}, [[EMPTY]], [[ARGS_ROOT_OF_KERNEL_1:![0-9]+]]}

// AIR-DAG: [[EMPTY]] = !{}

// AIR-DAG: [[ARGS_ROOT_OF_KERNEL_0]] = !{[[SHARED_DATA_ARG:![0-9]+]], [[SCALAR_ARG:![0-9]+]]}
// AIR-DAG: [[ARGS_ROOT_OF_KERNEL_1]] = !{[[SHARED_DATA_ARG]]}

// AIR-DAG: [[SHARED_DATA_ARG]] = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"data"}

// AIR-DAG: [[SCALAR_ARG]] = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"scalar_value"}

