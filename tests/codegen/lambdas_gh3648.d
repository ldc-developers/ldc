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

    (x) // template
    {
        __gshared int lambda_templ;
        return x + lambda_templ;
    }(123);
}

// the global variables should be defined as linkonce_odr:
// CHECK: _D14lambdas_gh364815__lambda_L5_C12FZ10global_bari{{.*}} = linkonce_odr {{(hidden )?}}thread_local global
// CHECK: _D14lambdas_gh364816__lambda_L10_C20FZ18global_bar_inlinedOi{{.*}} = linkonce_odr {{(hidden )?}}global
// CHECK: _D14lambdas_gh36483fooFZ__T15__lambda_L21_C5TiZQuFiZ12lambda_templi{{.*}} = linkonce_odr {{(hidden )?}}global

// foo() should only call two lambdas:
// CHECK: define {{.*}}_D14lambdas_gh36483fooFZv
// CHECK-NEXT: call {{.*}}__lambda_L5_C12
// CHECK-NEXT: call {{.*}}__T15__lambda_L21_C5
// CHECK-NEXT: ret void

// bar() should be defined as linkonce_odr:
// CHECK: define linkonce_odr {{.*}}__lambda_L5_C12

// bar_inlined() should NOT have made it to the .ll:
// CHECK-NOT: define {{.*}}__lambda_L10_C20

// the template lambda instance should be defined as linkonce_odr:
// CHECK: define linkonce_odr {{.*}}__T15__lambda_L21_C5
