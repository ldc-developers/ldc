// Test --ftime-trace functionality

// RUN: %ldc --ftime-trace --ftime-trace-file=%t.1 --ftime-trace-granularity=10 %s && FileCheck --check-prefix=ALL --check-prefix=FINE %s < %t.1
// RUN: %ldc --ftime-trace --ftime-trace-file=%t.2 --ftime-trace-granularity=20000 %s && FileCheck --check-prefix=ALL --check-prefix=COARSE %s < %t.2

// RUN: %ldc --ftime-trace --ftime-trace-file=- --ftime-trace-granularity=20000 %s | FileCheck --check-prefix=ALL --check-prefix=COARSE %s

// ALL: traceEvents

module ftimetrace;

// FINE: stdio
// FINE: ftime-trace.d:[[@LINE+1]]
import std.stdio;

int ctfe()
{
    int sum;
    foreach (i; 0..100)
    {
        sum += i;
    }
    return sum;
}

// COARSE-NOT: foo
// FINE: foo
// FINE: ftime-trace.d:[[@LINE+1]]
int foo()
{
    enum s = ctfe();
    return s;
}

void main()
{
    foo();
}

// ALL-DAG: ExecuteCompiler
// FINE-DAG: Linking executable
