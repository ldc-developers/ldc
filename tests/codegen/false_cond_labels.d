// Make sure that the dead code of an `if` (and `else if`) is elminated when the condition is constant
// _only_ if the block does not contain any labels.
// For example,
// int a = 1;
// if (false) {
// L1:
//   a = 2;
// } else {
//   goto L1;
// }
// assert(a == 2);

// RUN: %ldc -O0 -output-ll -of=%t.s %s && FileCheck %s < %t.s

// extern(C) to avoid name mangling.
extern(C) void foo()
{
    // CHECK: foo()
    // CHECK: %a = alloca i32, align 4
    // CHECK: br
    int a;
    if (0)
    {
L1:
        a = 1;
    }
    else
    {
        goto L1;
    }
}
