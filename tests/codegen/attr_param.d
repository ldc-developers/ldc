// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.attributes;

// CHECK:      define{{.*}} @{{.*}}3foo
// CHECK-SAME: i8*{{.*}} noalias %p_arg
void foo(@llvmAttr("noalias") void* p) {}

// CHECK:      define{{.*}} @{{.*}}3bar
// CHECK-SAME: float*{{.*}} noalias %data_arg
// CHECK-SAME: [16 x float]*{{.*}} noalias dereferenceable(64) %kernel
void bar(@restrict float* data, @restrict ref const float[16] kernel) {}

// CHECK:      define{{.*}} @{{.*}}14classReference
// CHECK-SAME: %object.Object*{{.*}} noalias %obj_arg
void classReference(@restrict Object obj) {}
