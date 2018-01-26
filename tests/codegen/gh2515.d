// For scope-allocated class objects, make sure the _d_callfinalizer()
// druntime call is elided if the object has no dtors and no monitor.

// RUN: %ldc -O3 -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import core.stdc.stdio : printf;

class Base
{
    int val = 123;
    void foo() { val *= 3; }
    void bar() { synchronized(this) val *= 2; }
}

class WithDtor : Base
{
    ~this() {}
}

class WithImplicitDtor : Base
{
    static struct S { int val; ~this() {} }
    S s;
}

// CHECK: define{{.*}} void @{{.*}}_D6gh251516noDtor_noMonitorFZv
void noDtor_noMonitor()
{
    scope b = new Base();
    b.foo();
    printf("%d\n", b.val);
    // CHECK-NOT: _d_callfinalizer
    // CHECK: ret void
}

// CHECK: define{{.*}} void @{{.*}}_D6gh251518noDtor_withMonitorFZv
void noDtor_withMonitor()
{
    scope b = new Base();
    b.bar();
    printf("%d\n", b.val);
    // CHECK: _d_callfinalizer
    // CHECK: ret void
}

// CHECK: define{{.*}} void @{{.*}}_D6gh25158withDtorFZv
void withDtor()
{
    scope Base b = new WithDtor();
    b.foo();
    printf("%d\n", b.val);
    // CHECK: _d_callfinalizer
    // CHECK: ret void
}

// CHECK: define{{.*}} void @{{.*}}_D6gh251516withImplicitDtorFZv
void withImplicitDtor()
{
    scope Base b = new WithImplicitDtor();
    b.foo();
    printf("%d\n", b.val);
    // CHECK: _d_callfinalizer
    // CHECK: ret void
}


/* Test a C++ class as well, which as of 2.077 isn't implicitly delete()d. */

extern(C++) class CppClass
{
    int val = 666;
}

// CHECK: define{{.*}} void @{{.*}}_D6gh25158cppClassFZv
void cppClass()
{
    scope c = new CppClass();
    printf("%d\n", c.val);
    // CHECK-NOT: _d_callfinalizer
    // CHECK: ret void
}
