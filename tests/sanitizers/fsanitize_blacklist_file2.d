// Test sanitizer blacklist file functionality
// This file is blacklisted (file1 is not)

// RUN: %ldc -c -output-ll -fsanitize=address \
// RUN: -fsanitize-blacklist=%S/inputs/fsanitize_blacklist_file.txt \
// RUN: -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK-LABEL: define {{.*}}9foofoofoo
// CHECK-SAME: #[[ATTR_NOASAN:[0-9]+]]
void foofoofoo(int* i)
{
    *i = 1;
}
//CHECK: attributes #[[ATTR_NOASAN]]
//CHECK-NOT: sanitize_address
