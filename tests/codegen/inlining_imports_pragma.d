// Test inlining of functions marked with pragma(inline) in an imported module

// O0 and O3 should behave the same for these tests with explicit inlining directives by the user.

// RUN: %ldc %s -I%S -c -output-ll -O0 -of=%t.O0.ll && FileCheck %s --check-prefix OPTNONE < %t.O0.ll
// RUN: %ldc %s -I%S -c -output-ll -O3 -of=%t.O3.ll && FileCheck %s --check-prefix OPT3 < %t.O3.ll

import inputs.inlinables;

extern (C): // simplify mangling for easier matching

// OPTNONE-LABEL: define{{.*}} @call_never_inline(
// OPT3-LABEL: define{{.*}} @call_never_inline(
int call_never_inline()
{
    // OPTNONE: call {{.*}} @never_inline()
    // OPT3: call {{.*}} @never_inline()
    return never_inline();
}
// OPTNONE-DAG: declare {{.*}} @never_inline()

// OPTNONE-LABEL: define{{.*}} @call_always_inline(
// OPT3-LABEL: define{{.*}} @call_always_inline(
int call_always_inline()
{
    // OPTNONE-NOT: call {{.*}} @always_inline()
    // OPT3-NOT: call {{.*}} @always_inline()
    return always_inline();
    // OPTNONE: ret
    // OPT3: ret
}

// OPTNONE-LABEL: define{{.*}} @call_inline_chain(
// OPT3-LABEL: define{{.*}} @call_inline_chain(
int call_inline_chain()
{
    // OPTNONE-NOT: call
    // OPT3-NOT: call
    return always_inline_chain0();
    // OPTNONE: ret
    // OPT3: ret
}
