// RUN: %ldc -O3 -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK:      define {{.*}}zeroext {{.*}}@{{.*}}_D6gh21313foo
// CHECK-SAME: i1 {{(inreg )?}}zeroext %x_arg
bool foo(bool x, ref bool o)
{
    // CHECK-NOT: and i8
    // CHECK:     load i8{{.*}}, !range ![[META:[0-9]+]]
    // CHECK-NOT: and i8
    o |= x;
    // CHECK:     ret
    return o;
}

// CHECK: ![[META]] = {{.*}}!{i8 0, i8 2}
