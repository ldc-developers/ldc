// Tests successfull musttail application

// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK-LABEL: define{{.*}} @{{.*}}foo
int foo(int x) nothrow
{
    // CHECK: musttail call{{.*}} @{{.*}}bar
    // CHECK-NEXT: ret i32
    pragma(musttail) return bar(x);
}

// CHECK-LABEL: define{{.*}} @{{.*}}bar
int bar(int x) nothrow { return x; }
