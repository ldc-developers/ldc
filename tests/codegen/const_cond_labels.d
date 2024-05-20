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

// Note that a label is conssidered anything that lets us jump inside the body
// of the statement _apart from_ the actual statement (e.g. the `if).
// That generally is a normal label, but also specific cases for switch
// statements (see last tests).

// RUN: %ldc -O0 -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

extern(C):  //to avoid name mangling.
// CHECK-LABEL: @foo
void foo()
{
    // CHECK: %a = alloca
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

// CHECK-LABEL: @bar
void bar(int a, int b)
{
    int c;
    if (0)
    {
        switch (a) {
        case 10:
            while (b) {
            L2:
                // CHECK: store i32 10, ptr %c
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

// CHECK-LABEL: @without_goto
void without_goto(int a, int b)
{
    int c;
    if (0)
    {
        switch (a) {
        case 10:
            while (b) {
            L2:
                // CHECK: store i32 10, ptr %c
                c = 10;
            }
        default: assert(0);
        }
    }
    else
    {
        a = 2;
    }
}

// CHECK-LABEL: @fourth
void fourth(int a, int b, int c)
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
                // CHECK: store i32 10, ptr %d
                d = 10;
            }
        }
    }
}

// If a switch is outside the body of a statement "to-be-elided"
// but a case statement of it is inside that body, then that
// case acts as a label because we can jump inside the body without
// using the statement (i.e. using the switch to jump to the case).

// CHECK-LABEL: @case_as_label
void case_as_label(int a, int b)
{
    // Note the `CHECK-NOT` trickery.
    // CHECK-NOT: store i32 2, ptr %c
    // CHECK: store i32 3, ptr %c
    // CHECK-NOT: store i32 2, ptr %c
    // CHECK: store i32 4, ptr %c
    // CHECK-NOT: store i32 2, ptr %c
    int c;
    switch (a) {
    case 1:
        // Can elide
        if (false) {
            final switch (b) {
            case 2:
                c = 2;   
            }
        }
        goto case;
    case 2:
        // Can't elide
        if (false) {
    case 3:
            c = 3;
        }
        // Can't elide
        if (false) {
    default:
            c = 4;
        }
    }
}

// CHECK-LABEL: @case_as_label2
void case_as_label2(int a, int b)
{
    // CHECK: store i32 2, ptr %c
    // CHECK: store i32 3, ptr %c
    int c;
    final switch (a) {
        // Can't elide
        if (false) {
            final switch (b) {
            case 2:
                c = 2;   
            }
    // Test that `switch` in higher or equal nesting level
    // with a `case` does not impact the handling of `case`s.
    case 1:
        c = 3;
        }
    }
}
