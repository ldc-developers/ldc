// RUN: %ldc -I%S -c -output-ll -O3 -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.attributes;

extern (C): // For easier name mangling

int glob1;
int easily_inlinable(int i)
{
    glob1 = i;
    return 2;
}

// CHECK-LABEL: define{{.*}} @call_easily_inlinable(
// CHECK-SAME: #[[OPTNONE:[0-9]+]]
@optStrategy("none")
int call_easily_inlinable(int i)
{
    // CHECK: call {{.*}} @easily_inlinable(
    return easily_inlinable(i);
}

pragma(inline, true) int always_inline()
{
    return 321;
}
// CHECK-LABEL: define{{.*}} @call_always_inline(
@optStrategy("none")
int call_always_inline()
{
    // CHECK-NOT: call {{.*}} @always_inline()
    return always_inline();
    // CHECK: ret i32 321
}

// optnone functions should not be inlined.
int glob2;
@optStrategy("none") void optnone_function(int i)
{
    glob2 = i;
}

// CHECK-LABEL: define{{.*}} @call_optnone(
void call_optnone()
{
    // CHECK: call {{.*}} @optnone_function
    optnone_function(1);
}

// CHECK-LABEL: define{{.*}} @foo(
// CHECK-SAME: #[[OPTSIZE:[0-9]+]]
@optStrategy("optsize")
void foo()
{
}

// CHECK-LABEL: define{{.*}} @foo2(
// CHECK-SAME: #[[MINSIZE:[0-9]+]]
@optStrategy("minsize")
void foo2()
{
}

// CHECK-DAG: attributes #[[OPTNONE]] = {{.*}} noinline {{.*}} optnone
// CHECK-DAG: attributes #[[OPTSIZE]] = {{.*}} optsize
// CHECK-DAG: attributes #[[MINSIZE]] = {{.*}} minsize
