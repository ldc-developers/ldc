// Test basic --ftime-trace functionality

// REQUIRES: atleast_llvm1000

// RUN: %ldc --ftime-trace --ftime-trace-file=%t.1 %s                                   && FileCheck --check-prefix=ALL --check-prefix=FINE %s < %t.1
// RUN: %ldc --ftime-trace --ftime-trace-file=%t.2 --ftime-trace-granularity=20000 %s && FileCheck --check-prefix=ALL --check-prefix=COARSE %s < %t.2

// ALL: traceEvents

module ftimetrace;

// COARSE-NOT: intrinsics
// FINE: intrinsics
// FINE: ftime-trace.d([[@LINE+1]])
import ldc.intrinsics;

// COARSE-NOT: foo
// FINE: foo
// FINE: ftime-trace.d([[@LINE+1]])
int foo()
{
    return 1;
}

void main()
{
    foo();
}

// ALL: ExecuteCompiler
// ALL: Linking executable
