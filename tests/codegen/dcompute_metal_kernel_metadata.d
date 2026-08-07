// REQUIRES: target_AArch64

// COM: There are certain breaking changes in llvm upstream that are not yet supported by MSL compiler 
// COM: hence it is required to run on past versions of llvm
// REQUIRES: atleast_llvm20
 
// RUN: %ldc -c -mdcompute-targets=metal-400 --mdcompute-file-prefix=kernel_metadata -output-ll -of=%t.ll %s
// RUN: FileCheck %s --check-prefix=AIR < kernel_metadata_metal400_64.ll
@compute(CompileFor.deviceOnly) module kernels;
import ldc.dcompute;

// AIR-LABEL: define {{.*}} @{{.*}}test_kernel{{.*}}(
@kernel()
void test_kernel(GlobalPointer!float data) {
    data[0] = 42.0;
}

// AIR-DAG: !air.kernel = !{[[KERNEL:![0-9]+]]}
// AIR-DAG: !air.version = !{[[AIR_VERSION:![0-9]+]]}
// AIR-DAG: !air.language_version = !{[[AIR_LANGUAGE_VERSION:![0-9]+]]}
// AIR-DAG: [[KERNEL]] = !{ptr @{{.*}}test_kernel{{.*}}, [[EMPTY:![0-9]+]], [[ARGS_ROOT:![0-9]+]]}

// AIR-DAG: [[EMPTY]] = !{}

// AIR-DAG: [[ARGS_ROOT]] = !{[[ARG_0:![0-9]+]]}

// AIR-DAG: [[ARG_0]] = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"data"}

// AIR-DAG: [[AIR_VERSION]] = !{i32 2, i32 8, i32 0}
// AIR-DAG: [[AIR_LANGUAGE_VERSION]] = !{!"Metal", i32 4, i32 0, i32 0}
