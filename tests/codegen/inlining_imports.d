// Test inlining of imported functions

// RUN: %ldc %s -I%S -c -output-ll -release -O3 -enable-cross-module-inlining -of=%t.O3.ll && FileCheck %s --check-prefix OPT3 < %t.O3.ll

import inputs.inlinables;

extern (C): // simplify mangling for easier matching

// Simple functions for reference.
int foo()
{
    return goo();
}

int goo()
{
    return 1;
}

// OPT3-LABEL: define{{.*}} @call_easily_inlinable(
int call_easily_inlinable(int i)
{
    // OPT3-NOT: call {{.*}} @easily_inlinable(
    return easily_inlinable(i);
    // OPT3: ret i32 2
}

// OPT3-LABEL: define{{.*}} @call_class_function(
int call_class_function(A a)
{
    // OPT3-NOT: call
    return a.final_class_function();
    // OPT3: ret i32 12345
}

// OPT3-LABEL: define{{.*}} @call_weak_function(
int call_weak_function()
{
    // OPT3: call
    return weak_function();
    // OPT3-NOT: 654
    // Test for function end `}` to prevent matching "654" elsewhere (e.g. the LDC version git hash)
    // OPT3: }
}
