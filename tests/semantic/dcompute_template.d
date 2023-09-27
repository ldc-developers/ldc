// RUN: %ldc -o- -mdcompute-targets=cuda-350 %s -I%S
// REQUIRES: target_NVPTX
@compute(CompileFor.deviceOnly) module tests.semaintic.dcompute_template;
import ldc.dcompute;
import inputs.notatcompute : identity;

@kernel void test()()
{
    auto x = identity(42);
}
alias realtest = test!();
