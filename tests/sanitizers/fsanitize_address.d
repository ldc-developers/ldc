// Test basic AddressSanitizer functionality

// RUN: %ldc -c -output-ll -g -fsanitize=address -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -c -output-ll -g -fsanitize=address,thread -of=%t.tsan.ll %s && FileCheck %s --check-prefix=wTSAN < %t.tsan.ll
// RUN: %ldc -c -output-ll -g -fsanitize=address -fsanitize=thread -of=%t.tsan.ll %s && FileCheck %s --check-prefix=wTSAN < %t.tsan.ll

void foo(size_t a) {
    // wTSAN: call {{.*}}_tsan_func_entry
    // CHECK: call {{.*}}_asan_stack_malloc
    // wTSAN: {{(call|invoke)}} {{.*}}_asan_stack_malloc
    int[10] i;
    // CHECK: call {{.*}}_asan_report_store
    // wTSAN: {{(call|invoke)}} {{.*}}_asan_report_store
    i[a] = 1;
}
