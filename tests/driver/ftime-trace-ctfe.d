// Test --ftime-trace functionality: CTFE

// RUN: %ldc -c --ftime-trace --ftime-trace-granularity=100 --ftime-trace-file=%t.1 %s && FileCheck %s < %t.1

// CHECK: traceEvents

// CHECK-DAG: CTFE func: mulmul
// CHECK-DAG: CTFE start: Thing
// CHECK-DAG: CTFE func: muloddmul
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


int times(int mul, int i){
    return mul * i;
}

int muloddmul(int i) {
    int mul = 666;
    for (; i > 0; i--)
    {
        mul = times(mul, i);
    }
    return mul;
}

struct Thing {
    int m;
    this(int i) {
        if (i % 2) {
            m = muloddmul(i);
        } else {
            m = mulmul(i);
        }
    }
}

static immutable Thing thing = Thing(123000);

int foo() {
    auto thing2 = new Thing(123001);
    return 1;
}

enum f = foo();
