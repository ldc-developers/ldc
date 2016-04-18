// Tests @llvmAttr attribute

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.attributes;

extern (C): // For easier name mangling

// CHECK: define{{.*}} @keyvalue{{.*}} #[[KEYVALUE:[0-9]+]]
@(llvmAttr("key", "value"))
void keyvalue()
{
}

// CHECK: define{{.*}} @keyonly{{.*}} #[[KEYONLY:[0-9]+]]
@(llvmAttr("keyonly"))
void keyonly()
{
}

// CHECK-DAG: attributes #[[KEYVALUE]] = {{.*}} "key"="value"
// CHECK-NOT: attributes #[[KEYONLY]] = {{.*}} "keyonly"=
// CHECK-DAG: attributes #[[KEYONLY]] = {{.*}} "keyonly"
