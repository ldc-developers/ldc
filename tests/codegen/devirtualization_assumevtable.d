// Tests that class member function calls do not prevent devirtualization (vtable cannot change in class member call).

// RUN: %ldc -output-ll -of=%t.ll %s -O3 && FileCheck %s < %t.ll


class A {
    void foo();
    void oof();
}
class B : A {
    override void foo();
    override void oof();
}

// CHECK-LABEL: define{{.*}}ggg
void ggg()
{
    A a = new A();
    // CHECK: call {{.*}}_D29devirtualization_assumevtable1A3foo
    a.foo();
    // CHECK: call {{.*}}_D29devirtualization_assumevtable1A3foo
    a.foo();
}

// CHECK-LABEL: define{{.*}}hhh
void hhh()
{
    A a = new A();
    // CHECK: call {{.*}}_D29devirtualization_assumevtable1A3foo
    a.foo();
    // CHECK: call {{.*}}_D29devirtualization_assumevtable1A3oof
    a.oof();
}

// CHECK-LABEL: define{{.*}}exacttypeunknown
void exacttypeunknown(A a, A b)
{
    // CHECK: %[[FOO:[0-9a-z]+]] = load {{.*}}"foo@vtbl
    // CHECK: call{{.*}} void %[[FOO]](
    a.foo();
    // CHECK: call{{.*}} void %[[FOO]](
    a.foo();

    a = b;
    // CHECK: %[[FOO2:[0-9a-z]+]] = load {{.*}}"foo@vtbl
    // CHECK: call{{.*}} void %[[FOO2]](
    a.foo();
}

// The devirtualization is not valid for C++ methods.
extern(C++)
class CPPClass {
    void foo();
    void oof();
}

// CHECK-LABEL: define{{.*}}exactCPPtypeunknown
void exactCPPtypeunknown(CPPClass a)
{
    // CHECK: %[[FOO:[0-9a-z]+]] = load {{.*}}!invariant
    // CHECK: call{{.*}} void %[[FOO]](
    a.foo();
    // CHECK: %[[FOO2:[0-9a-z]+]] = load {{.*}}!invariant
    // CHECK: call{{.*}} void %[[FOO2]](
    a.foo();
}
