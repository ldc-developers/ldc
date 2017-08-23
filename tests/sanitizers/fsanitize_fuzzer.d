// Test basic Fuzz sanitizer functionality

// REQUIRES: atleast_llvm400

// RUN: %ldc -c -output-ll -O3 -fsanitize=fuzzer -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -c -output-ll -fsanitize=fuzzer,address -of=%t.asan.ll %s && FileCheck %s --check-prefix=wASAN < %t.asan.ll

// CHECK-LABEL: define{{.*}} @{{.*}}FuzzMe
// wASAN-LABEL: define{{.*}} @{{.*}}FuzzMe
bool FuzzMe(const ubyte* data, size_t dataSize)
{
    // CHECK: call {{.*}}_sanitizer_cov_trace_pc_guard
    // CHECK: call {{.*}}_sanitizer_cov_trace_{{(const_)?}}cmp

    return dataSize >= 3 &&
           data[0] == 'F' &&
           data[1] == 'U' &&
           data[2] == 'Z' &&
           data[3] == 'Z'; // :‑<
}

// CHECK-LABEL: define{{.*}} @{{.*}}allocInt
// wASAN-LABEL: define{{.*}} @{{.*}}allocInt
void allocInt() {
    // CHECK: call {{.*}}_sanitizer_cov_trace_pc_guard
    // wASAN: call {{.*}}_asan_stack_malloc
    int[10] a;
}

// CHECK-LABEL: define{{.*}} @{{.*}}foo
// wASAN-LABEL: define{{.*}} @{{.*}}foo
void foo(int function() a) {
    // CHECK: call {{.*}}_sanitizer_cov_trace_pc_guard
    // CHECK: call void @__sanitizer_cov_trace_pc_indir
    a();
}
