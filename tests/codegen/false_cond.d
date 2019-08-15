// Make sure that the `else`/`else if` _will_ be generated if the `if` condition has a constant 0 value.

// RUN: %ldc -O0 -output-ll -of=%t.s %s && FileCheck %s < %t.s

// extern(C) to avoid name mangling.
extern(C) void foo()
{
    // CHECK: foo()
    // CHECK-NOT: %a = alloca i32, align 4
    // CHECK: %b = alloca i32, align 4
    // CHECK-NOT: br
    // CHECK-NOT: store i32 1, i32* %a
    // CHECK: store i32 2, i32* %b
    if (0)
    {
        int a = 1;
    }
    else
    {
        int b = 2;
    }
}

extern(C) void bar()
{
    // CHECK: bar()
    // CHECK-NOT: %a = alloca i32, align 4
    // CHECK: %b = alloca i32, align 4
    // CHECK-NOT: br
    // CHECK-NOT: store i32 1, i32* %a
    // CHECK: store i32 2, i32* %b
    if (0)
    {
        int a = 1;
    }
    else if(1)
    {
        int b = 2;
    }
}

extern(C) void only_ret()
{
    // CHECK: only_ret
    // CHECK-NEXT: ret void
    // CHECK-NEXT: }
    if (1 && (2 - 2))
    {
        int a = 1;
    }
}

extern(C) void only_ret2()
{
    // CHECK: only_ret2
    // CHECK-NEXT: ret void
    // CHECK-NEXT: }
    if (0)
    {
        int a = 1;
    }
    else if(0)
    {
        int b = 2;
    }
}

extern(C) void gen_br(immutable int a)
{
    // CHECK: gen_br(i32 %
    // CHECK-COUNT-1: br
    if (a)
    {
        int b = 1;
    }
}
