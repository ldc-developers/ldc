// Test -fno-delete-null-pointer-checks

// RUN: %ldc -c -output-ll -of=%t_ub.ll %s && FileCheck -check-prefix=NULL_UB -check-prefix=BOTH %s < %t_ub.ll
// RUN: %ldc -fno-delete-null-pointer-checks -c -output-ll -of=%t.ll %s && FileCheck -check-prefix=NULL_OK -check-prefix=BOTH %s < %t.ll

// BOTH-LABEL: define{{.*}} @foo
// BOTH-SAME: #[[ATTR:[0-9]+]]
extern (C) double foo(double a, double b)
{
    double c;

    return a * b + c;

// BOTH-LABEL: define{{.*}} @{{.*}}nested_func
// NULL_UB-SAME: nonnull
// NULL_OK-NOT: nonnull
    void nested_func()
    {
        c = 1;
    }
}

struct S
{
// BOTH-LABEL: define{{.*}} @{{.*}}member_func
// NULL_UB-SAME: nonnull
// NULL_OK-NOT: nonnull
    void member_func() {}
}

class C
{
// BOTH-LABEL: define{{.*}} @{{.*}}member_2_func
// NULL_UB-SAME: nonnull
// NULL_OK-NOT: nonnull
    void member_2_func() {}
}

// BOTH-LABEL: define{{.*}} @{{.*}}some_other_function
// NULL_UB-SAME: nonnull
// NULL_OK-NOT: nonnull
struct Opaque;
void some_other_function(ref Opaque b)
{

}

// NULL_OK-DAG: attributes #[[ATTR]] ={{.*null.pointer.is.valid}}
// NULL_UB-NOT: null-pointer-is-valid
// NULL_UB-NOT: null_pointer_is_valid
