// Test sanitizer blacklist file functionality
// This file is blacklisted (file1 is not)

// RUN: %ldc -c -output-ll -fsanitize=address \
// RUN: -fsanitize-blacklist=%S/inputs/fsanitize_blacklist_file.txt \
// RUN: -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK-NOT: ; Function Attrs:{{.*}} sanitize_address
// CHECK:     define{{.*}} void {{.*}}9foofoofoo
void foofoofoo(int* i)
{
    *i = 1;
}
