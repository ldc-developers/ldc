// Tests that lambdas and contained globals are emitted as linkonce_odr.

// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

enum bar = ()
{
    static int global_bar;
};

enum bar_inlined = ()
{
    pragma(inline, true);
    shared static int global_bar_inlined;
};

void foo()
{
    bar();
    bar_inlined();
}

// the global variables should be defined as linkonce_odr:
// CHECK: _D14lambdas_gh36489__lambda5FZ10global_bari{{.*}} = linkonce_odr thread_local global
// CHECK: _D14lambdas_gh36489__lambda6FZ18global_bar_inlinedOi{{.*}} = linkonce_odr global

// foo() should only call one lambda:
// CHECK: define {{.*}}_D14lambdas_gh36483fooFZv
// CHECK-NEXT: call {{.*}}__lambda
// CHECK-NEXT: ret void

// bar() should be defined as linkonce_odr:
// CHECK: define linkonce_odr {{.*}}__lambda

// bar_inlined() should NOT have made it to the .ll:
// CHECK-NOT: define {{.*}}__lambda
