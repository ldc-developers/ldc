// RUN: not %ldc -o- -mdcompute-targets=cuda-350 -I%S %s 2>&1 | FileCheck %s
// REQUIRES: target_NVPTX
@compute(CompileFor.deviceOnly) module tests.semaintic.dcompute_template_fail;
import ldc.dcompute;
import inputs.notatcompute : callsSomeFunc;

//CHECK: inputs/notatcompute.d(9): Error: can only call functions from other `@compute` modules in `@compute` code
@kernel void test()
{
    callsSomeFunc();
}
