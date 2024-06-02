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

// CHECK-DAG: _D{{.*}}1C7__ClassZ{{\"?}} = global %object.TypeInfo_Class
// CHECK-DAG: _D{{.*}}classvarC14TypeInfo_Class{{\"?}} = thread_local global ptr {{.*}}1C7__ClassZ
auto classvar = typeid(C);

// CHECK-DAG: _D{{.*}}TypeInfo_C{{.*}}1I6__initZ{{\"?}} = linkonce_odr global %object.TypeInfo_Interface
// CHECK-DAG: _D{{.*}}interfacevarC18TypeInfo_Interface{{\"?}} = thread_local global ptr {{.*}}TypeInfo_C{{.*}}1I6__initZ
auto interfacevar = typeid(I);

// CHECK-DAG: _D{{.*}}TypeInfo_S{{.*}}1S6__initZ{{\"?}} = linkonce_odr global %object.TypeInfo_Struct
// CHECK-DAG: _D{{.*}}structvarC15TypeInfo_Struct{{\"?}} = thread_local global ptr {{.*}}TypeInfo_S{{.*}}1S6__initZ
auto structvar = typeid(S);
