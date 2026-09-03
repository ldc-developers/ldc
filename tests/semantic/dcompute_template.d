// RUN: %ldc -o- -mdcompute-targets=cuda-350 %s -I%S
// REQUIRES: target_NVPTX
@compute(CompileFor.deviceOnly) module tests.semantic.dcompute_template;
import ldc.dcompute;
import inputs.notatcompute : identity, A, B;

@kernel void test()()
{
    auto x = identity(42);
}
alias realtest = test!();

@kernel void test2()
{
    auto x = identity(42);
}

@kernel void test3()
{
    A a;
    a.foo();
}

@kernel void test4()
{
    B!() b;
    b.foo();
}

