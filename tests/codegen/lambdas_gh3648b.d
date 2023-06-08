// Tests that *imported* lambdas and contained globals are emitted with internal linkage.

// RUN: %ldc -output-ll -of=%t.ll %s -I%S && FileCheck %s < %t.ll

import lambdas_gh3648;

void foo()
{
    bar();
    bar_inlined();
}

// the global variables should be defined as internal:
// CHECK: _D14lambdas_gh36489__lambda5FZ10global_bari{{.*}} = internal thread_local global
// CHECK: _D14lambdas_gh36489__lambda6FZ18global_bar_inlinedOi{{.*}} = internal global

// foo() should only call one lambda:
// CHECK: define {{.*}}_D15lambdas_gh3648b3fooFZv
// CHECK-NEXT: call {{.*}}__lambda5
// CHECK-NEXT: ret void

// bar() should be defined as internal:
// CHECK: define internal {{.*}}__lambda5

// bar_inlined() should NOT have made it to the .ll:
// CHECK-NOT: define {{.*}}__lambda6
