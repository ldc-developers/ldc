// Check that we ignore the conversion of `assert(x)` -> `assert(x,_d_assert_fail(...))`
// and dont give any errors for the generated code (which is ignored by codegen).
// REQUIRES: target_NVPTX
// RUN: %ldc -c -mdcompute-targets=cuda-350 -m64 -output-ll -mdcompute-file-prefix=check_action -output-o -checkaction=context
// xRUN: FileCheck %s --check-prefix=LL < dcompute_check_action.ll

@compute(CompileFor.deviceOnly) module dcompute_check_action;

import ldc.dcompute;

@kernel void test()
{
    if(__dcompute_reflect(ReflectTarget.CUDA,0))
    {
    }
    else
        assert(0);
}
