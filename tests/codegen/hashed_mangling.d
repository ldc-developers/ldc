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
    // HASH90-DAG: define{{.*}} @{{(\"\\01_?)?}}_D3one3two5three__T1sTiZQfFNaNbNiNfiZSQBkQBjQBi__TQBfTiZQBlFiZ__T6ResultTiZQk
    // HASH90-DAG: define{{.*}} @{{(\"\\01_?)?}}_D3one3two5three3L1633_182fab6f09ff014d9f4a578edf9609981sZ
    // HASH90-DAG: define{{.*}} @{{(\"\\01_?)?}}_D3one3two5three3L2333_9b5306e5c42722cd2cb93ae6beb422346Result3fooZ
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
        // HASH90-DAG: define{{.*}} @{{(\"\\01_?)?}}_D3one3two5three__T5klassTiZQjFiZ__T6ResultTiZQk3fooMFZv
        // HASH90-DAG: define{{.*}} @{{(\"\\01_?)?}}_D3one3two5three3L3433_de737f3d65ae58efa925cffda52cd8da6Result3fooZ
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
