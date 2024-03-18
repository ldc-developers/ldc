// Test that inlining does not leak definitions without marking them as available_externally
// "Leaking" = symbols definitions in .o file that shouldn't be declarations instead (undefined symbols).

// RUN: %ldc %s -I%S -c -output-ll -release                  -O3 -enable-cross-module-inlining -of=%t.O3.ll && FileCheck %s --check-prefix OPT3 < %t.O3.ll
// RUN: %ldc %s -I%S -c -output-ll -release -enable-inlining -O0 -enable-cross-module-inlining -of=%t.O0.ll && FileCheck %s --check-prefix OPT0 < %t.O0.ll
// RUN: %ldc -I%S -enable-inlining -enable-cross-module-inlining %S/inputs/inlinables.d -run %s
// RUN: %ldc -I%S -O3 -enable-cross-module-inlining %S/inputs/inlinables.d -run %s

import inputs.inlinables;

extern (C): // simplify mangling for easier matching

// Check that the global variables that are added due to "available_externally
// inlining" do not have initializers, i.e. they are declared only and not definined.

// OPT3-DAG: @module_variable = external thread_local{{.*}} global i32, align
// OPT3-DAG: @{{.*}}write_function_static_variableUiZ15static_func_vari{{\"?}} = external thread_local{{.*}} global i32, align

// OPT0-LABEL: define{{.*}} @call_class_function(
// OPT3-LABEL: define{{.*}} @call_class_function(
int call_class_function(A a)
{
    // There should be only one call to "virtual_func".
    // OPT3: call
    // OPT3-NOT: call
    return a.final_func();
    // There should be a return from an LLVM variable (not a direct value)
    // OPT0: ret i32 %
    // OPT3: ret i32 %
}

// OPT0-LABEL: define{{.*}} @dont_leak_module_variables(
// OPT3-LABEL: define{{.*}} @dont_leak_module_variables(
void dont_leak_module_variables()
{
    write_module_variable(987);
    write_function_static_variable(167);
    get_typeid_A();
    // OPT0: ret void
    // OPT3: ret void
}

// OPT0-LABEL: define{{.*}} @main(
// OPT3-LABEL: define{{.*}} @main(
int main()
{
    dont_leak_module_variables();

    return 0;
    // OPT0: ret i32 0
    // OPT3: ret i32 0
}
