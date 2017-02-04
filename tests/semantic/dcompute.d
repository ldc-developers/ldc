// RUN: not %ldc -o- -I%S %s 2>&1 | FileCheck %s


@compute(CompileFor.deviceOnly) module dcompute;
import ldc.attributes;
import inputs.notatcompute : somefunc;

extern(C) bool perhaps();
extern(C) bool __dcompute_reflect(int,int);
//CH ECK: Error: interfaces and classes not allowed in @compute code
interface I {}

//CH ECK: Error: interfaces and classes not allowed in @compute code
class C : Throwable { this() { super(""); } }

//CHECK: Error: {{.*}} global variables not allowed in @compute code
C c;

void func()
{
    //CHECK: Error: {{.*}} associative arrays not allowed in @compute code
    int[int] foo;
    //CHECK: Error: array literal in @compute code not allowed
    auto bar = [0, 1, 2];
    //CHECK: Error: cannot use 'new' in @compute code
    auto baz = new int;
    //CHECK: Error: cannot use 'delete' in @compute code
    delete baz;

    int[] quux;
    //CHECK: Error: setting 'length' in @compute code not allowed
    quux.length = 1;
    //CHECK: Error: cannot use operator ~= in @compute code
    quux ~= 42;
    //CHECK: Error: cannot use operator ~ in @compute code
    cast(void) (quux ~ 1);
    //CHECK: Error: typeinfo not available in @compute code
    cast(void) typeid(int);
    //CHECK: Error: cannot use 'synchronized' in @compute code
    synchronized {}
    //CHECK: Error: string literals not allowed in @compue code
    auto s = "geaxsese";
    //CHECK: Error: cannot switch on strings in @compute code
    switch(s)
    {
        default:
            break;
    }

    //CHECK: Error: can only call functions from other @compute modules in @compute code
    somefunc();
    if (__dcompute_reflect(0,0))
        //CHECK-NOT: Error: can only call functions from other @compute modules in @compute code
        somefunc();

    //CHECK: Error: no exceptions in @compute code
    try
    {
        func1();
    }
    catch(C c)
    {
    }

    if (perhaps())
        //CHECK: Error: no exceptions in @compute code
        throw c;

    //CHECK-NOT: Error: no exceptions in @compute code
    try
    {
        func1();
    }
    finally
    {
        func2();
    }
    //CHECK-NOT: Error: no exceptions in @compute code
    scope(exit)
        func2();

    //CHECK: Error: asm not allowed in @compute code
    asm {ret;}
}

void func1() {}
void func2() {}

//CH ECK: Error: linking additional libraries not supported in @compute code
pragma(lib, "bar");
