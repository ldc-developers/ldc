// Test sanitizer blacklist file functionality
// This file is _not_ blacklisted (file2 is)

// RUN: %ldc -c -output-ll -fsanitize=address \
// RUN: -fsanitize-blacklist=%S/inputs/fsanitize_blacklist_file.txt \
// RUN: -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK-LABEL: define {{.*}}9foofoofoo
// CHECK-SAME: #[[ATTR_WITHASAN:[0-9]+]]
void foofoofoo(int* i)
{
    // CHECK: call {{.*}}_asan
    *i = 1;
}

//CHECK: attributes #[[ATTR_WITHASAN]] ={{.*}}sanitize_address
