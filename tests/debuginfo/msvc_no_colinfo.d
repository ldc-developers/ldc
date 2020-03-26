// REQUIRES: Windows
// RUN: %ldc -g -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

void foo(int a)
{
    a += 3;
}

// CHECK-NOT: column:
// CHECK: !DILocation(line:
// CHECK-NOT: column:
