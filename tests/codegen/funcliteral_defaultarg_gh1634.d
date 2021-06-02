// Test function literal as default argument

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -run %s

module mod;

// CHECK-LABEL: define{{.*}} @{{.*}}D3mod3fooFPFZiZi
int foo(int function() d = () { return 123; })
{
    return d();
}

// CHECK-LABEL: define{{.*}} @{{.*}}D3mod8call_fooFZi
int call_foo()
{
    // CHECK: call {{.*}}D3mod3fooFPFZiZi{{.*}}D3mod9__lambda5FNaNbNiNfZi
    return foo();
}

// The lambda is defined by the first call to foo with default arguments.
// CHECK-LABEL: define{{.*}} @{{.*}}D3mod9__lambda5FNaNbNiNfZi
// CHECK: ret i32 123

// CHECK-LABEL: define{{.*}} @{{.*}}Dmain
void main()
{
    // CHECK: call {{.*}}D3mod3fooFPFZiZi{{.*}}D3mod9__lambda5FNaNbNiNfZi
    assert(foo() == 123);
}
