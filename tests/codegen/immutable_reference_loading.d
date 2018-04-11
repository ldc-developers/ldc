// Test that immutable references are only loaded once.

// RUN: %ldc -O3 -release -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

extern (C): // simplify mangling

int clobber(); // opaque call

// First test that we correctly recognize the double load
// CHECK-LABEL: define{{.*}} @const_ptr
auto const_ptr(const int* a, int* b)
{
    // CHECK: = load
    auto temp = *a;
    *b = 42;
    // CHECK: = load
    return temp + *a;
    // CHECK: ret
}

// CHECK-LABEL: define{{.*}} @immut_ptr
auto immut_ptr(immutable int* a, int* b)
{
    // CHECK: = load
    auto temp = *a;
    *b = 42;
    // CHECK-NOT: = load
    return temp + *a;
    // CHECK: ret
}

// CHECK-LABEL: define{{.*}} @mutable_ptr_to_immut
auto mutable_ptr_to_immut(immutable(int)* a, int* b)
{
    // CHECK: = load
    auto temp = *a;
    *b = 42;
    // CHECK-NOT: = load
    return temp + *a;
    // CHECK: ret
}

// CHECK-LABEL: define{{.*}} @mutate_ptr
auto mutate_ptr(immutable(int)* a, int* b, immutable(int)* c)
{
    // CHECK: = load
    auto temp = *a;
    a = c;
    // CHECK: = load
    return temp + *a;
    // CHECK: ret
}

// CHECK-LABEL: define{{.*}} @immut_ref
auto immut_ref(ref immutable float a)
{
    // CHECK: = load
    auto temp = a;
    clobber();
    // CHECK-NOT: = load
    return temp + a;
    // CHECK: ret
}

// CHECK-LABEL: define{{.*}} @immutptr_ref
auto immutptr_ref(ref immutable float* a)
{
    // CHECK: = load
    auto temp = *a;
    clobber();
    // CHECK: = load
    return temp + *a;
    // CHECK: ret
}

// Test byval parameter attribute (caller passes pointer to private allocated block of mem)
// CHECK-LABEL: define{{.*}} @immut_byvalarray
auto immut_byvalarray(int[100] a)
{
    // CHECK: = load
    auto temp = a[0];
    clobber();
    // CHECK-NOT: = load
    return temp + a[0];
    // CHECK: ret
}

// CHECK-LABEL: define{{.*}} @inout_ptr
auto inout_ptr(inout long* a)
{
    // CHECK: = load
    auto temp = *a;
    clobber();
    // CHECK: = load
    return temp + *a;
    // CHECK: ret
}

// CHECK-LABEL: define{{.*}} @templ_call
auto templ_call(immutable(int)* a)
{
    // CHECK: = load
    // CHECK-NOT: = load
    // CHECK: ret
    return templ_ptr(a);
}

// CHECK-LABEL: define{{.*}} @{{.*}}9templ_ptr
auto templ_ptr(T)(T* a)
{
    // CHECK: = load
    auto temp = *a;
    clobber();
    // CHECK-NOT: = load
    return temp + *a;
    // CHECK: ret
}

// CHECK-LABEL: define{{.*}} @immut_cast
auto immut_cast(int* a)
{
    // CHECK: = load
    auto temp = *a;
    clobber();
    // CHECK: = load
    temp += templ_ptr(cast(immutable)a);
    // `noalias` shouldn't leak into this function after inlining the above call, so we need to load from the pointer again below.
    // CHECK: = load
    temp += *a;
    // CHECK: ret
    return temp;
}

class A {
    int i;
}

// CHECK-LABEL: define{{.*}} @immut_class
auto immut_class(immutable(A) a)
{
    // CHECK: = load
    auto temp = a.i;
    clobber();
    // CHECK-NOT: = load
    return temp + a.i;
    // CHECK: ret
}
