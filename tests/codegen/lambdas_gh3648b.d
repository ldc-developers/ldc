// Tests that *imported* lambdas and contained globals are emitted as linkonce_odr.

// RUN: %ldc -output-ll -of=%t.ll %s -I%S && FileCheck %s < %t.ll

import lambdas_gh3648;

void foo()
{
    bar();
    bar_inlined();
}

// the global variables should be defined as linkonce_odr:
// CHECK: _D14lambdas_gh36489__lambda5FZ10global_bari{{.*}} = linkonce_odr {{(hidden )?}}thread_local global
// CHECK: _D14lambdas_gh36489__lambda6FZ18global_bar_inlinedOi{{.*}} = linkonce_odr {{(hidden )?}}global

// foo() should only call one lambda:
// CHECK: define {{.*}}_D15lambdas_gh3648b3fooFZv
// CHECK-NEXT: call {{.*}}__lambda5
// CHECK-NEXT: ret void

// bar() should be defined as linkonce_odr:
// CHECK: define linkonce_odr {{.*}}__lambda5

// bar_inlined() should NOT have made it to the .ll:
// CHECK-NOT: define {{.*}}__lambda6
