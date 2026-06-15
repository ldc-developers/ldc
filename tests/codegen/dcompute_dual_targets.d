// Check that we can generate code for both the host and multiple devices in one compiler invocation
// REQUIRES: target_NVPTX
// REQUIRES: target_SPIRV
// RUN: %ldc -c -mdcompute-targets=cuda-350,ocl-220 -m64 -output-ll -mdcompute-file-prefix=dual -I%S/inputs -output-o %s %S/inputs/kernel.d
// RUN: FileCheck %s --check-prefix=LL < dcompute_dual_targets.ll

import inputs.kernel : k_foo;
import ldc.dcompute;

void main(string[] args)
{
    string s = foo.mangleof;
    string k_s = k_foo.mangleof;

    GlobalPointer!float global_x;
    foo(global_x);
}

// LL-DAG: __dcompute_ptx_internal_{{.*}} align 4
// LL-DAG: __dcompute_ptx_{{.*}} ={{.*}}alias

// LL-DAG: __dcompute_spv_internal_{{.*}} align 4
// LL-DAG: __dcompute_spv_{{.*}} ={{.*}}alias

void foo(GlobalPointer!float x_in) {
    SharedPointer!float shared_x;
    *shared_x = *x_in;
}
