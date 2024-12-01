// Tests that *imported* lambdas and contained globals are emitted with internal linkage.

// RUN: %ldc -output-ll -of=%t.ll %s -I%S && FileCheck %s < %t.ll

import lambdas_gh3648;

void foo()
{
    bar();
    bar_inlined();
}

// the global variables should be defined as internal:
// CHECK: _D14lambdas_gh364815__lambda_L5_C12FZ10global_bari{{.*}} = internal thread_local global
// CHECK: _D14lambdas_gh364816__lambda_L10_C20FZ18global_bar_inlinedOi{{.*}} = internal global

// foo() should only call one lambda:
// CHECK: define {{.*}}_D15lambdas_gh3648b3fooFZv
// CHECK-NEXT: call {{.*}}__lambda_L5_C12
// CHECK-NEXT: ret void

// bar() should be defined as internal:
// CHECK: define internal {{.*}}__lambda_L5_C12

// bar_inlined() should NOT have made it to the .ll:
// CHECK-NOT: define {{.*}}__lambda_L10_C20
