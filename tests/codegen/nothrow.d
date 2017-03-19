// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

struct S
{
    ~this() nothrow {}
    void foo() nothrow { throw new Error("foo"); }
}

struct Throwing
{
    ~this() {}
    void bar() { throw new Exception("bar"); }
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D7nothrow15inTryCatchErrorFZv
void inTryCatchError()
{
    try
    {
        // make sure the nothrow functions S.foo() and S.~this()
        // are invoked in try-blocks with at least 1 catch block
        S a;
        // CHECK: invoke {{.*}}_D7nothrow1S3fooMFNbZv{{.*}} %a
        a.foo();
        // CHECK: invoke {{.*}}_D7nothrow1S6__dtorMFNbZv{{.*}} %a
    }
    catch (Error) {}
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D7nothrow19inTryCatchExceptionFZv
void inTryCatchException()
{
    // make sure the nothrow functions are never invoked
    // CHECK-NOT: invoke {{.*}}_D7nothrow1S3fooMFNbZv
    // CHECK-NOT: invoke {{.*}}_D7nothrow1S6__dtorMFNbZv

    try
    {
        S a;
        a.foo();
    }
    catch (Exception) {}
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D7nothrow12inTryFinallyFZv
void inTryFinally()
{
    // make sure the nothrow functions are never invoked
    // CHECK-NOT: invoke {{.*}}_D7nothrow1S3fooMFNbZv
    // CHECK-NOT: invoke {{.*}}_D7nothrow1S6__dtorMFNbZv

    try
    {
        S a;
        a.foo();
    }
    finally
    {
        S b;
        b.foo();
    }
}

// CHECK-LABEL: define{{.*}} @{{.*}}_Dmain
void main()
{
    // make sure the nothrow functions are never invoked
    // CHECK-NOT: invoke {{.*}}_D7nothrow1S3fooMFNbZv
    // CHECK-NOT: invoke {{.*}}_D7nothrow1S6__dtorMFNbZv

    Throwing t;

    S a;
    a.foo();
    t.bar();

    {
        S b;
        t.bar();
        b.foo();

        S().foo();
    }
}
