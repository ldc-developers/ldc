// Test cross-module inlining involving static variables

// RUN: %ldc %s -I%S -c -output-ll                  -O3 -of=%t.O3.ll && FileCheck %s --check-prefix OPT3 < %t.O3.ll
// RUN: %ldc %s -I%S -c -output-ll -enable-inlining -O0 -of=%t.O0.ll && FileCheck %s --check-prefix OPT0 < %t.O0.ll
// RUN: %ldc -I%S -enable-inlining %S/inputs/inlinables_staticvar.d -run %s
// RUN: %ldc -I%S -O3              %S/inputs/inlinables_staticvar.d -run %s

import inputs.inlinables_staticvar;

import ldc.attributes;

extern (C): // simplify mangling for easier matching

// Functions are intentionally split and @weak to thwart LLVM constant folding.

void checkModuleScope_1() @weak
{
    addToModuleScopeInline(7);
}
void checkModuleScope_2() @weak
{
    addToModuleScopeOutline(101);
    assert(equalModuleScope(7+101));
}

void checkInsideFunc_1() @weak
{
    assert(addAndCheckInsideFunc(0, 7));
}
void checkInsideFunc_2() @weak
{
    assert(addAndCheckInsideFuncIndirect(7, 101));
    assert(addAndCheckInsideFunc(7+101, 9));
}

void checkInsideNestedFunc_1() @weak
{
    assert(addAndCheckInsideNestedFunc(0, 7));
}
void checkInsideNestedFunc_2() @weak
{
    assert(addAndCheckInsideNestedFuncIndirect(7, 101));
    assert(addAndCheckInsideNestedFunc(7+101, 9));
}

void checkNestedStruct_1() @weak
{
    assert(addAndCheckNestedStruct(0, 7));
}
void checkNestedStruct_2() @weak
{
    assert(addAndCheckNestedStructIndirect(7, 101));
    assert(addAndCheckNestedStruct(7+101, 9));
}

// OPT0-LABEL: define{{.*}} @_Dmain(
// OPT3-LABEL: define{{.*}} @_Dmain(
extern(D)
void main()
{
    checkModuleScope_1();
    checkModuleScope_2();
    checkInsideFunc_1();
    checkInsideFunc_2();
    checkInsideNestedFunc_1();
    checkInsideNestedFunc_2();
    checkNestedStruct_1();
    checkNestedStruct_2();
}
