// Test sanitizer blacklist functionality

// RUN: %ldc -c -output-ll -fsanitize=address \
// RUN: -fsanitize-blacklist=%S/inputs/fsanitize_blacklist.txt \
// RUN: -fsanitize-blacklist=%S/inputs/fsanitize_blacklist2.txt \
// RUN: -of=%t.ll %s && FileCheck %s < %t.ll

// Don't attempt to load the blacklist when no sanitizer is active
// RUN: %ldc -o- -fsanitize-blacklist=%S/thisfilecertainlydoesnotexist %s

// CHECK-LABEL: define {{.*}}9foofoofoo
// CHECK-SAME: #[[ATTR_WITHASAN:[0-9]+]]
void foofoofoo(int* i)
{
    // CHECK: call {{.*}}_asan
    *i = 1;
}

// CHECK-LABEL: define {{.*}}blacklisted
// CHECK-SAME: #[[ATTR_NOASAN:[0-9]+]]
extern (C) void blacklisted(int* i)
{
    // CHECK-NOT: call {{.*}}_asan
    *i = 1;
}

// Test blacklisted wildcard
// CHECK-LABEL: define {{.*}}10black_set1
// CHECK-SAME: #[[ATTR_NOASAN:[0-9]+]]
void black_set1(int* i)
{
    // CHECK-NOT: call {{.*}}_asan
    *i = 1;
}
// CHECK-LABEL: define {{.*}}10black_set2
// CHECK-SAME: #[[ATTR_NOASAN:[0-9]+]]
void black_set2(int* i)
{
    // CHECK-NOT: call {{.*}}_asan
    *i = 1;
}

//  Test blacklisting of template class methods
class ABCDEF(T)
{
    void method(int* i)
    {
        *i = 1;
    }
}

// CHECK-LABEL: define {{.*}}__T6ABCDEFTiZQk6method
// CHECK-SAME: #[[ATTR_NOASAN:[0-9]+]]
ABCDEF!int ofBlacklistedType;

// CHECK-LABEL: define {{.*}}__T6ABCDEFTAyaZQm6method
// CHECK-SAME: #[[ATTR_WITHASAN:[0-9]+]]
ABCDEF!string ofInstrumentedType;

//CHECK: attributes #[[ATTR_WITHASAN]] ={{.*}}sanitize_address
//CHECK: attributes #[[ATTR_NOASAN]]
//CHECK-NOT: sanitize_address
//CHECK-SAME: }
