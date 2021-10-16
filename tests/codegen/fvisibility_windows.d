// Windows: test generated IR for -fvisibility / `export` / @hidden combinations.

// REQUIRES: Windows

// RUN: %ldc -fvisibility=public -output-ll -of=%t_public.ll %s
// RUN: FileCheck %s --check-prefix=PUBLIC --check-prefix=COMMON < %t_public.ll

// RUN: %ldc -fvisibility=hidden -output-ll -of=%t_hidden.ll %s
// RUN: FileCheck %s --check-prefix=HIDDEN --check-prefix=COMMON < %t_hidden.ll

import ldc.attributes : hidden;

extern(C):

export
{
    // COMMON-DAG: @exportedGlobal = dllexport global i32
    __gshared int exportedGlobal;
    // COMMON-DAG: define dllexport void @exportedFunc()
    void exportedFunc() {}
}

// PUBLIC-DAG: @global = dllexport global i32
// HIDDEN-DAG: @global = global i32
__gshared int global;
// PUBLIC-DAG: define dllexport void @func()
// HIDDEN-DAG: define void @func()
void func() {}

@hidden
{
    // COMMON-DAG: @hiddenGlobal = global i32
    __gshared int hiddenGlobal;
    // COMMON-DAG: define void @hiddenFunc()
    void hiddenFunc() {}

    export
    {
        // COMMON-DAG: @exportedHiddenGlobal = dllexport global i32
        __gshared int exportedHiddenGlobal;
        // COMMON-DAG: define dllexport void @exportedHiddenFunc()
        void exportedHiddenFunc() {}
    }
}
