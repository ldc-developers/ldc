// RUN: %ldc -betterC -run %s > %t.stdout
// RUN: FileCheck %s < %t.stdout

import core.stdc.stdio;

void notNothrow() { printf("notNothrow\n"); }

struct WithDtor
{
    ~this() { printf("destructing\n"); }
}

void foo(WithDtor a, bool skip)
{
    if (skip)
        return;

    notNothrow();
}

extern(C) int main()
{
    WithDtor a;
    scope(exit) printf("exiting\n");

    foo(a, true);
    // CHECK:      destructing

    foo(a, false);
    // CHECK-NEXT: notNothrow
    // CHECK-NEXT: destructing

    return 0;
    // CHECK-NEXT: exiting
    // CHECK-NEXT: destructing
}
