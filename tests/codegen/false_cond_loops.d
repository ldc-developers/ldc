// Make sure that if a `while`/`for` condition has a constant 0 value, then don't generate any code for this block.

// RUN: %ldc -O0 -output-ll -of=%t.s %s && FileCheck %s < %t.s

// extern(C) to avoid name mangling.
extern(C) void foo()
{
    // CHECK: foo()
    // CHECK-NOT: br
    // Note: `const` and `immutable` work. They resolve to constant values only through
    // semantic analysis. Using a normal local variable, even if not mutated, it
    // is required data-flow analysis for this to be confirmed (or for the final value
    // to be known at compile-time).
    immutable int a = 0;
    const int b = a && 1;
    while (b)
    {
        int c = 1 + 2;
    }

    for (0)
    {
        int d = 1;
    }
}
