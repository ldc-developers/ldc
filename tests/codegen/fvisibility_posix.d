// Non-Windows: test generated IR for -fvisibility / `export` / @hidden combinations.

// UNSUPPORTED: Windows

// RUN: %ldc -fvisibility=public -output-ll -of=%t_public.ll %s
// RUN: FileCheck %s --check-prefix=PUBLIC --check-prefix=COMMON < %t_public.ll

// RUN: %ldc -fvisibility=hidden -output-ll -of=%t_hidden.ll %s
// RUN: FileCheck %s --check-prefix=HIDDEN --check-prefix=COMMON < %t_hidden.ll

import ldc.attributes : hidden;

extern(C):

export
{
    // COMMON-DAG: @exportedGlobal = global i32
    __gshared int exportedGlobal;
    // COMMON-DAG: define void @exportedFunc()
    void exportedFunc() {}
}

// PUBLIC-DAG: @global = global i32
// HIDDEN-DAG: @global = hidden global i32
__gshared int global;
// PUBLIC-DAG: define void @func()
// HIDDEN-DAG: define hidden void @func()
void func() {}

@hidden
{
    // COMMON-DAG: @hiddenGlobal = hidden global i32
    __gshared int hiddenGlobal;
    // COMMON-DAG: define hidden void @hiddenFunc()
    void hiddenFunc() {}

    export
    {
        // COMMON-DAG: @exportedHiddenGlobal = global i32
        __gshared int exportedHiddenGlobal;
        // COMMON-DAG: define void @exportedHiddenFunc()
        void exportedHiddenFunc() {}
    }
}
