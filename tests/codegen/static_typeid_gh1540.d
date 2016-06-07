// Tests correct codegen for static variables initialized with typeid(A)
// Test for Github issue 1540

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

class C
{
}

interface I
{
}

struct S
{
}

// CHECK: @{{.*}}classvarC14TypeInfo_Class = thread_local global %object.TypeInfo_Class* @{{.*}}1C7__ClassZ
auto classvar = typeid(C);

// CHECK: @{{.*}}interfacevarC18TypeInfo_Interface = thread_local global %"typeid({{.*}}.I)"* @_D{{[0-9]+}}TypeInfo_C{{.*}}1I6__initZ
auto interfacevar = typeid(I);

// CHECK: @{{.*}}structvarC15TypeInfo_Struct = thread_local global %"typeid({{.*}}S)"* @_D{{[0-9]+}}TypeInfo_S{{.*}}1S6__initZ
auto structvar = typeid(S);
