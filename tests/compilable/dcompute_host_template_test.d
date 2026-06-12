// RUN: %ldc -c -mdcompute-targets=cuda-350 -I%S %s
@compute(CompileFor.deviceOnly) module dcompute_host_template_test;
import ldc.dcompute;
import inputs.host_template_module;

void kernel() {
    // Instantiate a template that resides in a hostOnly module.
    // This should compile successfully because DCompute semantic analysis
    // should skip over host-only template instantiations.
    alias X = HostTemplate!int;
}
