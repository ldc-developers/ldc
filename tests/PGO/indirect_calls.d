// Test instrumentation of indirect calls

// REQUIRES: PGO_RT

// FIXME: fails with LLVM 13+, call remains indirect
// XFAIL: *

// RUN: %ldc -c -output-ll -fprofile-instr-generate -of=%t.ll %s && FileCheck %s --check-prefix=PROFGEN < %t.ll

// RUN: %ldc -fprofile-instr-generate=%t.profraw -run %s  \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -O3 -c -output-ll -of=%t2.ll -fprofile-instr-use=%t.profdata %s \
// RUN:   &&  FileCheck %s -check-prefix=PROFUSE < %t2.ll

import ldc.attributes;

extern (C)
{ // simplify name mangling for simpler string matching

    @optStrategy("none") // don't inline / elide call
    void hot()
    {
    }

    void luke()
    {
    }

    void cold()
    {
    }

    void function() foo;

    @weak // disable reasoning about this function
    void select_func(int i)
    {
        if (i < 1700)
            foo = &hot;
        else if (i < 1990)
            foo = &luke;
        else
            foo = &cold;
    }

} // extern C

// PROFGEN-LABEL: @_Dmain(
// PROFUSE-LABEL: @_Dmain(
int main()
{
    for (int i; i < 2000; ++i)
    {
        select_func(i);

        // PROFGEN:  [[REG1:%[0-9]+]] = load void ()*, void ()** @foo
        // PROFGEN-NEXT:  [[REG2:%[0-9]+]] = ptrtoint void ()* [[REG1]] to i64
        // PROFGEN-NEXT:  call void @__llvm_profile_instrument_target(i64 [[REG2]], i8* bitcast ({{.*}}_Dmain to i8*), i32 0)
        // PROFGEN-NEXT:  call void [[REG1]]()

        // PROFUSE:  [[REG1:%[0-9]+]] = load void ()*, void ()** @foo
        // PROFUSE:  [[REG2:%[0-9]+]] = icmp eq void ()* [[REG1]], @hot
        // PROFUSE:  call void @hot()
        // PROFUSE:  call void [[REG1]]()

        foo();
    }

    return 0;
}
