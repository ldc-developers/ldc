// RUN: %ldc -c -output-ll -fno-builtin -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK: define {{.*}} @builtin{{.*}} [[ATTR:#[0-9]+]]
extern(C) void builtin() {}

// CHECK: attributes [[ATTR]] = { {{.*}}"no-built-in"{{.*}} }