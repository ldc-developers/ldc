// Test instrumentation of indirect calls

// REQUIRES: PGO_RT

// FIXME: fails with LLVM 13+ for Windows, call remains indirect
// XFAIL: Windows

// RUN: %ldc -O3 -fprofile-generate=%t.profraw -run %s  \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -O3 -c -output-ll -of=%t.use.ll -fprofile-use=%t.profdata %s \
// RUN:   &&  FileCheck %s -check-prefix=PROFUSE < %t.use.ll

import ldc.attributes;

extern (C)
{ // simplify name mangling for simpler string matching

    @optStrategy("none")
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

// PROFUSE-LABEL: @_Dmain(
int main()
{
    for (int i; i < 2000; ++i)
    {
        select_func(i);

        // PROFUSE:  [[REG1:%[0-9]+]] = load ptr, ptr @foo
        // PROFUSE:  [[REG2:%[0-9]+]] = icmp eq ptr [[REG1]], @hot
        // PROFUSE:  call void @hot()
        // PROFUSE:  call void [[REG1]]()

        foo();
    }

    return 0;
}
