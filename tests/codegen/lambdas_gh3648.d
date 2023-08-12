// Tests that lambdas and contained globals are emitted with internal linkage.

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

    (x) // template
    {
        __gshared int lambda_templ;
        return x + lambda_templ;
    }(123);
}

// the global variables should be defined as internal:
// CHECK: _D14lambdas_gh36489__lambda5FZ10global_bari{{.*}} = internal thread_local global
// CHECK: _D14lambdas_gh36489__lambda6FZ18global_bar_inlinedOi{{.*}} = internal global
// CHECK: _D14lambdas_gh36483fooFZ__T9__lambda1TiZQnFiZ12lambda_templi{{.*}} = internal global

// foo() should only call two lambdas:
// CHECK: define {{.*}}_D14lambdas_gh36483fooFZv
// CHECK-NEXT: call {{.*}}__lambda5
// CHECK-NEXT: call {{.*}}__T9__lambda1
// CHECK-NEXT: ret void

// bar() should be defined as internal:
// CHECK: define internal {{.*}}__lambda5

// bar_inlined() should NOT have made it to the .ll:
// CHECK-NOT: define {{.*}}__lambda6

// the template lambda instance should be defined as internal:
// CHECK: define internal {{.*}}__T9__lambda1
