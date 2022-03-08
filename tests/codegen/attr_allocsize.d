// Test ldc.attributes.allocSize

// RUN: %ldc -O3 -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.attributes;

// CHECK-LABEL: define{{.*}}@{{.*}}my_calloc
// CHECK-SAME: #[[ATTR0:[0-9]+]]
extern (C) void* my_calloc(size_t num, size_t size) @allocSize(1, 0)
{
    return null;
}

// CHECK-LABEL: define{{.*}}@{{.*}}my_malloc
// CHECK-SAME: #[[ATTR1:[0-9]+]]
extern (C) void* my_malloc(int a, int b, size_t size, int c) @allocSize(2)
{
    return null;
}

// Test the reversed parameter order of D calling convention
// CHECK-LABEL: define{{.*}}@{{.*}}Dlinkage_calloc
// CHECK-SAME: #[[ATTR2:[0-9]+]]
extern (D) void* Dlinkage_calloc(int size, int b, size_t num, int c) @allocSize(0, 2)
{
    return null;
}

// Test function type with hidden `this` argument
class A
{
    // CHECK-LABEL: define{{.*}}@{{.*}}this_calloc
    // CHECK-SAME: #[[ATTR3:[0-9]+]]
    void* this_calloc(int size, int b, size_t num, int c) @allocSize(0, 2)
    {
        return null;
    }
}

// CHECK-DAG: attributes #[[ATTR0]] ={{.*}} allocsize(1,0)
// CHECK-DAG: attributes #[[ATTR1]] ={{.*}} allocsize(2)
// CHECK-DAG: attributes #[[ATTR2]] ={{.*}} allocsize(0,2)
// CHECK-DAG: attributes #[[ATTR3]] ={{.*}} allocsize(1,3)
