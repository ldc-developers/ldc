// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.attributes;

// CHECK:      define{{.*}} @{{.*}}3foo
// CHECK-SAME: i8*{{.*}} noalias align 1 %p_arg
void foo(@llvmAttr("noalias") void* p) {}

// CHECK:      define{{.*}} @{{.*}}3bar
// CHECK-SAME: [16 x float]*{{.*}} noalias align 4 dereferenceable(64) %kernel
// CHECK-SAME: float*{{.*}} noalias align 4 %data_arg
void bar(@restrict float* data, @restrict ref const float[16] kernel) {}

// CHECK:      define{{.*}} @{{.*}}14classReference
// CHECK-SAME: %object.Object*{{.*}} noalias %obj_arg
void classReference(@restrict Object obj) {}
