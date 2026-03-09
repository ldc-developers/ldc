// RUN: %ldc -c -output-ll -fsanitize=fuzzer -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK: define {{.*}} @f{{.*}} [[ATTR:#[0-9]+]]
extern(C) void f() {}

// CHECK: attributes [[ATTR]] = { {{.*}}optforfuzzing{{.*}} }
