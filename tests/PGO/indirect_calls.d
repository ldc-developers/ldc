// Test instrumentation of C-style indirect calls

// RUN: %ldc -c -output-ll -fprofile-instr-generate -fprofile-indirect-calls -of=%t.ll %s && FileCheck %s --check-prefix=PROFGEN < %t.ll

// RUN: %ldc -fprofile-instr-generate=%t.profraw -fprofile-indirect-calls -run %s  \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -O3 -c -output-ll -of=%t2.ll -fprofile-instr-use=%t.profdata -fprofile-indirect-calls %s \
// RUN:   &&  FileCheck %s -check-prefix=PROFUSE < %t2.ll

import ldc.attributes : weak;

extern (C)
{ // simplify name mangling for simpler string matching

    @weak // disable inlining of this function
    void hot()
    {
       // LLVM missing feature: need capability to add symbols to pointer lookup table!
       // pragma(LDC_profile_instr, false);
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


// ICP without profiling should generate same IR
pragma(LDC_profile_instr, false)
{
auto is_likely(alias A, Fptr, Args...)(Fptr fptr, Args args)
{
    return (fptr == &A) ? A(args) : fptr(args);
}

int manual_optimization()
{
    for (int i; i < 2000; ++i)
    {
        select_func(i);

        foo.is_likely!hot();
    }

    return 0;
}
}
