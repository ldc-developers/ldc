// Tests that array (in)equality with null is optimized to a length check

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -run %s

// CHECK-LABEL: define{{.*}} @{{.*}}isNull
bool isNull(int[] arg)
{
    // CHECK-NOT: call
    // CHECK-NOT: invoke
    // CHECK: icmp eq i{{32|64}} %.len, 0
    return arg == null;
}

// CHECK-LABEL: define{{.*}} @{{.*}}isNotNull
bool isNotNull(int[] arg)
{
    // CHECK-NOT: call
    // CHECK-NOT: invoke
    // CHECK: icmp ne i{{32|64}} %.len, 0
    return arg != null;
}

void main()
{
    int[3] i3 = [ 0, 1, 2 ];
    assert(isNull(i3[0..0]));
    assert(isNotNull(i3[0..1]));
}
