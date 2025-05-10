// Test sanitizer blacklist file functionality
// This file is _not_ blacklisted (file2 is)

// RUN: %ldc -c -output-ll -fsanitize=address \
// RUN: -fsanitize-blacklist=%S/inputs/fsanitize_blacklist_file.txt \
// RUN: -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK:      ; Function Attrs:{{.*}} sanitize_address
// CHECK-NEXT: define{{.*}} void {{.*}}9foofoofoo
void foofoofoo(int* i)
{
    // CHECK: call {{.*}}_asan
    *i = 1;
}
