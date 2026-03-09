// RUN: %ldc -c -output-ll -fno-builtin -O %s -of=%t.ll && FileCheck %s < %t.ll

extern(C) void* memcpy(void* s1, const(void)* s2, size_t n);

// CHECK-LABEL: define {{.*}} @test_memcpy
extern(C) void test_memcpy(void* dst, void* src)
{
    // CHECK: call {{.*}} @memcpy
    memcpy(dst, src, 16);
}

// CHECK: attributes #{{[0-9]+}} = { {{.*}}"no-builtins"{{.*}} }
