// Test inlining of functions marked with pragma(inline)

// RUN: %ldc %s -I%S -c -output-ll -O0 -of=%t.O0.ll && FileCheck %s --check-prefix OPTNONE < %t.O0.ll
// RUN: %ldc %s -I%S -c -output-ll -O3 -of=%t.O3.ll && FileCheck %s --check-prefix OPT3 < %t.O3.ll

extern (C): // simplify mangling for easier matching

int dummy;

// OPTNONE-LABEL: define{{.*}} @never_inline
// OPTNONE-SAME: #[[NEVER:[0-9]+]]
pragma(inline, false) int never_inline()
{
    dummy = 111;
    return 222;
}

int external();

// OPTNONE-LABEL: define{{.*}} @always_inline
// OPTNONE-SAME: #[[ALWAYS:[0-9]+]]
pragma(inline, true) int always_inline()
{
    int a;
    foreach (i; 1 .. 10)
    {
        foreach (ii; 1 .. 10)
        {
            foreach (iii; 1 .. 10)
            {
                a += i * external();
            }
        }
    }
    dummy = 444;
    return a;
}

// OPTNONE-LABEL: define{{.*}} @foo
// OPTNONE-SAME: #[[FOO:[0-9]+]]
int foo()
{
    return 333;
}

// OPT3-LABEL: define{{.*}} @call_always_inline
int call_always_inline()
{
    // OPT3-NOT: call {{.*}} @always_inline()
    // OPT3: ret
    return always_inline();
}

// OPT3-LABEL: define{{.*}} @call_never_inline
int call_never_inline()
{
    // OPT3: call {{.*}} @never_inline()
    // OPT3: ret
    return never_inline();
}

// OPTNONE-NOT: attributes #[[FOO]] ={{.*}} alwaysinline
// OPTNONE-NOT: attributes #[[FOO]] ={{.*}} noinline
// OPTNONE-NOT: attributes #[[NEVER]] ={{.*}} alwaysinline
// OPTNONE-NOT: attributes #[[ALWAYS]] ={{.*}} noinline
// OPTNONE-DAG: attributes #[[NEVER]] ={{.*}} noinline
// OPTNONE-DAG: attributes #[[ALWAYS]] ={{.*}} alwaysinline
