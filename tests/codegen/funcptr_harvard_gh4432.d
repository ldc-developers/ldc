// Tests function pointers/delegates on a Harvard architecture,
// with code residing in a separate address space.

// REQUIRES: target_AVR
// RUN: %ldc -mtriple=avr -betterC -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -mtriple=avr -betterC -c %s

alias FP = void function();
alias DG = void delegate();

// CHECK: @_D22funcptr_harvard_gh44328globalFPPFZv = global ptr addrspace(1) @_D22funcptr_harvard_gh44323barFZv, align 2
__gshared FP globalFP = &bar;
// CHECK: @_D22funcptr_harvard_gh443217globalDataPointerPPFZv = global ptr @_D22funcptr_harvard_gh44328globalFPPFZv, align 2
__gshared FP* globalDataPointer = &globalFP;

// CHECK: define void @_D22funcptr_harvard_gh44323fooFPFZvDQeZv({{.*}} addrspace(1){{\*?}} %fp_arg, { {{.*}} addrspace(1){{\*?}} } %dg_arg) addrspace(1)
void foo(FP fp, DG dg)
{
    // CHECK: call addrspace(1) void %1()
    fp();
    // CHECK: call addrspace(1) void %.funcptr
    dg();
    // CHECK-NEXT: ret void
}

void bar()
{
    foo(() {}, delegate() {});

    FP fp = &bar;
    DG dg;
    dg.funcptr = &bar;
    foo(fp, dg);

    dg.funcptr = *globalDataPointer;
    foo(globalFP, dg);
}
