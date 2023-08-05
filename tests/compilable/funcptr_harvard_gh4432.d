// A minimal test for function pointers/delegates on a Harvard architecture,
// with code residing in a separate address space.

// REQUIRES: target_AVR
// RUN: %ldc -mtriple=avr -betterC -c %s

alias FP = void function();
alias DG = void delegate();

void foo(FP fp, DG dg)
{
    fp();
    dg();
}

void bar()
{
    foo(() {}, delegate() {});

    FP fp = &bar;
    DG dg;
    dg.funcptr = &bar;
    foo(fp, dg);
}
