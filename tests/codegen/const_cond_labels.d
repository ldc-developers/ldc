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

extern(C) void bar(int a, int b)
{
    int c;
    if (0)
    {
        switch (a) {
        case 10:
            while (b) {
            L2:
            // CHECK: store i32 10, i32* %c
                c = 10;
            }
        default: assert(0);
        }
    }
    else
    {
        goto L2;
    }
}

extern(C) void third(int a, int b, int c)
{
    int j, d;
    if (a)
    {
        goto L3;
    }
    else if (false)
    {
        for (j = 0; j <= c; ++j)
        {
            // Can't `goto` in a foreach because
            // it always declares a variable.
            L3:
            foreach (i; 0..b)
            {
                // CHECK: store i32 10, i32* %d
                d = 10;
            }
        }
    }
}
