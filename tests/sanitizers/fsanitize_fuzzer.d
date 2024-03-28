// Test basic Fuzz sanitizer functionality

// RUN: %ldc -c -output-ll -O3 -fsanitize=fuzzer -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -c -output-ll -fsanitize=fuzzer,address -of=%t.asan.ll %s && FileCheck %s --check-prefix=CHECK --check-prefix=wASAN < %t.asan.ll

// CHECK-LABEL: define{{.*}} @{{.*}}FuzzMe
bool FuzzMe(const ubyte* data, size_t dataSize)
{
    // CHECK: call {{.*}}_sanitizer_cov_trace_{{(const_)?}}cmp

    return dataSize >= 3 &&
           data[0] == 'F' &&
           data[1] == 'U' &&
           data[2] == 'Z' &&
           data[3] == 'Z'; // :‑<
}

// CHECK-LABEL: define{{.*}} @{{.*}}allocInt
void allocInt(size_t i) {
    // wASAN: call {{.*}}_asan_stack_malloc
    int[10] a;
    a[i] = 1;
}

// CHECK-LABEL: define{{.*}} @{{.*}}foo
void foo(int function() a) {
    // CHECK: call void @__sanitizer_cov_trace_pc_indir
    a();
}
