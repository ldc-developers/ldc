// Test basic coverage sanitizer functionality

// RUN: %ldc -c -output-ll -fsanitize-coverage=trace-pc-guard -of=%t.ll %s && FileCheck %s --check-prefix=PCGUARD < %t.ll
// RUN: %ldc -c -output-ll -fsanitize-coverage=trace-pc-guard,trace-cmp -of=%t.cmp.ll %s && FileCheck %s --check-prefix=PCCMP < %t.cmp.ll
// RUN: %ldc -c -output-ll -fsanitize-coverage=trace-pc-guard,func -of=%t.func.ll %s && FileCheck %s --check-prefix=PCFUNC < %t.func.ll

// PCGUARD-LABEL: define{{.*}} @{{.*}}FuzzMe
// PCCMP-LABEL: define{{.*}} @{{.*}}FuzzMe
// PCFUNC-LABEL: define{{.*}} @{{.*}}FuzzMe
bool FuzzMe(const ubyte* data, size_t dataSize)
{
    // PCGUARD: call {{.*}}_sanitizer_cov_trace_pc_guard
    // PCGUARD-NOT: call {{.*}}_sanitizer_cov_trace_{{(const_)?}}cmp
    // PCGUARD: call {{.*}}_sanitizer_cov_trace_pc_guard
    // PCGUARD-NOT: call {{.*}}_sanitizer_cov_trace_{{(const_)?}}cmp

    // PCCMP: call {{.*}}_sanitizer_cov_trace_pc_guard
    // PCCMP: call {{.*}}_sanitizer_cov_trace_{{(const_)?}}cmp

    // PCFUNC: call {{.*}}_sanitizer_cov_trace_pc_guard
    // PCFUNC-NOT: call {{.*}}_sanitizer_cov_trace_pc_guard

    return dataSize >= 3 &&
           data[0] == 'F' &&
           data[1] == 'U' &&
           data[2] == 'Z' &&
           data[3] == 'Z'; // :‑<
    // PCFUNC: ret i1
}
