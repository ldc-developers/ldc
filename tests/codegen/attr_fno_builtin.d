// RUN: %ldc -c -output-ll -fno-builtin -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK-LABEL: define {{.*}} @test_builtin(
// CHECK: call {{.*}} @printf
extern(C) int printf(const char*, ...);

extern(C) void test_builtin() {
    printf("hi reviewer\n");
}

// CHECK: attributes #{{[0-9]+}} = { {{.*}}"no-builtins"{{.*}} }
