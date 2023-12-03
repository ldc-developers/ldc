// RUN: not %ldc -o- -verrors=0 -I%S %s 2>&1 | FileCheck %s


@compute(CompileFor.deviceOnly) module tests.semaintic.dcompute;
import ldc.dcompute;
import inputs.notatcompute : somefunc;

extern(C) bool perhaps();
//CHECK: dcompute.d([[@LINE+1]]): Error: interfaces and classes not allowed in `@compute` code
interface I {}

//CHECK: dcompute.d([[@LINE+1]]): Error: interfaces and classes not allowed in `@compute` code
class C : Throwable { this() { super(""); } }

//CHECK: dcompute.d([[@LINE+1]]): Error: global variables not allowed in `@compute` code
C c;

void func()
{
    //CHECK: dcompute.d([[@LINE+1]]): Error: associative arrays not allowed in `@compute` code
    int[int] foo;
    //CHECK: dcompute.d([[@LINE+1]]): Error: array literal in `@compute` code not allowed
    auto bar = [0, 1, 2];
    //CHECK: dcompute.d([[@LINE+1]]): Error: cannot use `new` in `@compute` code
    auto baz = new int;

    //CHECK: dcompute.d([[@LINE+1]]): Error: interfaces and classes not allowed in `@compute` code
    I i;
    //CHECK: dcompute.d([[@LINE+1]]): Error: interfaces and classes not allowed in `@compute` code
    C cc;
    int[] quux;
    //CHECK: dcompute.d([[@LINE+1]]): Error: setting `length` in `@compute` code not allowed
    quux.length = 1;
    //CHECK: dcompute.d([[@LINE+1]]): Error: cannot use operator `~=` in `@compute` code
    quux ~= 42;
    //CHECK: dcompute.d([[@LINE+1]]): Error: cannot use operator `~` in `@compute` code
    cast(void) (quux ~ 1);
    //CHECK: dcompute.d([[@LINE+1]]): Error: typeinfo not available in `@compute` code
    cast(void) typeid(int);
    //CHECK: dcompute.d([[@LINE+1]]): Error: cannot use `synchronized` in `@compute` code
    synchronized {}
    //CHECK: dcompute.d([[@LINE+1]]): Error: string literals not allowed in `@compute` code
    auto s = "geaxsese";
    //CHECK: dcompute.d([[@LINE+1]]): Error: cannot `switch` on strings in `@compute` code
    switch(s)
    {
        default:
            break;
    }

    //CHECK: dcompute.d([[@LINE+1]]): Error: can only call functions from other `@compute` modules in `@compute` code
    somefunc();
    if (__dcompute_reflect(ReflectTarget.Host,0))
        //CHECK-NOT: Error:
        somefunc();

    //CHECK: dcompute.d([[@LINE+1]]): Error: no exceptions in `@compute` code
    try
    {
        func1();
    }
    catch(C c)
    {
    }

    if (perhaps())
        //CHECK: dcompute.d([[@LINE+1]]): Error: no exceptions in `@compute` code
        throw c;

    //CHECK-NOT: Error:
    try
    {
        func1();
    }
    finally
    {
        func2();
    }
    //CHECK-NOT: Error:
    scope(exit)
        func2();
}

void func1() {}
void func2() {}

//CHECK: dcompute.d([[@LINE+1]]): Error: linking additional libraries not supported in `@compute` code
pragma(lib, "bar");
