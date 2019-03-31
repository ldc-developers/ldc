// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.attributes;

// CHECK: define{{.*}} @{{.*}}3foo{{.*}}(i8* noalias %p_arg)
void foo(@llvmAttr("noalias") void* p) {}
