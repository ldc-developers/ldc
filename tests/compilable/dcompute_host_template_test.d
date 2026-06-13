// RUN: %ldc -c -mdcompute-targets=cuda-350 -I%S %s
// RUN: not %ldc -o- -mdcompute-targets=cuda-350 -verrors=0 -I%S -d-version=Fail %s 2>&1 | FileCheck %s

@compute(CompileFor.deviceOnly) module dcompute_host_template_test;
import ldc.dcompute;
import inputs.host_template_module;

void kernel() {
    // Instantiate a template that resides in a hostOnly module.
    // This should compile successfully because DCompute semantic analysis
    // should skip over host-only template instantiations.
    alias X = HostTemplate!int;

    version(Fail) {
        // CHECK: dcompute_host_template_test.d([[@LINE+1]]): Error: can only call functions from other `@compute` modules in `@compute` code
        X.doHostThings();
    }
}
