// Test -ffast-math

// RUN: %ldc -ffast-math -O0 -release -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.attributes;

// CHECK-LABEL: define{{.*}} @foo
// CHECK-SAME: #[[ATTR:[0-9]+]]
extern (C) double foo(double a, double b)
{
    // CHECK: fmul fast
    return a * b;
}

// CHECK-DAG: attributes #[[ATTR]] ={{.*}} "unsafe-fp-math"="true"
