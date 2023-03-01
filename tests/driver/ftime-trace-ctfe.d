// Test --ftime-trace functionality: CTFE

// RUN: %ldc -c --ftime-trace --ftime-trace-file=%t.1 %s && FileCheck %s < %t.1

// CHECK: traceEvents

// CHECK-DAG: CTFE start: Thing
// CHECK-DAG: CTFE call: mulmul
// CHECK-DAG: ExecuteCompiler

module ftimetrace;

int mulmul(int i) {
    int mul = 666;
    for (; i > 0; i--)
    {
        mul *= i;
    }
    return mul;
}

struct Thing {
    int m;
    this(int i) {
        m = mulmul(i);
    }
}

static immutable Thing thing = Thing(123000);
