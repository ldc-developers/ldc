// Make sure that the dead code of an `if` (and `else if`) is elminated when the condition is constant.
// If the condition value is constant, there are two cases:
// 1) It is 0 (false)    -> Generate the else block (if it exists) with no branching.
// 2) It is non-0 (true) -> Generate the if block with no branching.
// Also, verify it _does_ generate correct code when it is not constant.

// RUN: %ldc -O0 -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -O0 -run %s
// RUN: %ldc -O3 -run %s

extern(C):  //to avoid name mangling.

void g();

// CHECK-LABEL: @foo
void foo()
{
    // CHECK-NOT: %a = alloca
    // CHECK: %b = alloca
    // CHECK-NOT: br
    // CHECK-NOT: store i32 1, {{i32\*|ptr}} %a
    // CHECK: store i32 2, {{i32\*|ptr}} %b
    if (0)
    {
        int a = 1;
    }
    else
    {
        int b = 2;
    }
}

// CHECK-LABEL: @bar
void bar()
{
    // CHECK-NOT: %a = alloca
    // CHECK: store i32 2, {{i32\*|ptr}} %b
    if (0)
    {
        int a = 1;
    }
    else if(1)
    {
        int b = 2;
    }
}

// CHECK-LABEL: @only_ret
void only_ret()
{
    // CHECK-NEXT: ret void
    // CHECK-NEXT: }
    if (1 && (2 - 2))
    {
        int a = 1;
    }
}

// CHECK-LABEL: @only_ret2
void only_ret2()
{
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

// CHECK-LABEL: @only_ret3
void only_ret3()
{
    // CHECK-NEXT: ret void
    // CHECK-NEXT: }
    if (!cast(void*)&only_ret2)
    {
        int a = 1;
    }
}

// CHECK-LABEL: @gen_br
void gen_br(immutable int a)
{
    // CHECK-COUNT-1: br
    if (a)
    {
        int b = 1;
    }
}

// CHECK-LABEL: @gh4556
void gh4556()
{
    if ("a" is "b")
    {
        assert(false);
    }

    if ("a" !is "b")
    {
        return;
    }
    else
    {
        assert(false);
    }
}

void main() {
    gh4556();
}
