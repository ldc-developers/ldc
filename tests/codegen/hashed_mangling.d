// Test hashing of symbols above hash threshold

// RUN: %ldc -hash-threshold=90 -g -c -output-ll -of=%t90.ll %s && FileCheck %s --check-prefix HASH90 < %t90.ll
// RUN: %ldc -hash-threshold=90 -run %s

// Don't use Phobos functions in this test, because the test hashthreshold is too low for an unhashed libphobos.

module one.two.three;

// HASH90-DAG: define{{.*}} @externCfunctions_are_not_hashed_externCfunctions_are_not_hashed_externCfunctions_are_not_hashed
extern (C) int externCfunctions_are_not_hashed_externCfunctions_are_not_hashed_externCfunctions_are_not_hashed()
{
    return 95;
}

auto s(T)(T t)
{
    // HASH90-DAG: define{{.*}} @{{(\"\\01_)?}}_D3one3two5three8__T1sTiZ1sFNaNbNiNfiZS3one3two5three8__T1sTiZ1sFiZ13__T6ResultTiZ6Result
    // HASH90-DAG: define{{.*}} @{{(\"\\01_)?}}_D3one3two5three3L1633_699ccf279a146992d539ca3ca16e22e11sZ
    // HASH90-DAG: define{{.*}} @{{(\"\\01_)?}}_D3one3two5three3L2333_5ee632e10b6f09e8f541a143266bdf226Result3fooZ
    struct Result(T)
    {
        void foo(){}
    }
    return Result!int();
}

auto klass(T)(T t)
{
    class Result(T)
    {
        // HASH90-DAG: define{{.*}} @{{(\"\\01_)?}}_D3one3two5three12__T5klassTiZ5klassFiZ13__T6ResultTiZ6Result3fooMFZv
        // HASH90-DAG: define{{.*}} @{{(\"\\01_)?}}_D3one3two5three3L3433_46a82aac733d8a4b3588d7fa8937aad66Result3fooZ
        void foo(){}
    }
    return new Result!int();
}

void main()
{
    assert(
        externCfunctions_are_not_hashed_externCfunctions_are_not_hashed_externCfunctions_are_not_hashed() == 95);

    auto x = 1.s.s.s.s;
    x.foo;

    auto y = 1.klass.klass.klass.klass;
    y.foo;
}
