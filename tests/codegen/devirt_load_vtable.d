// RUN: %ldc -c -output-ll -O3 -of=%t.ll %s && FileCheck %s < %t.ll

// LLVM versions before 3.9 do not perform all optizations tested here.
// REQUIRES: atleast_llvm309

class A
{
    void foo();
    int one()
    {
        return 1;
    }
}

// CHECK-LABEL: define{{.*}} @{{.*}}testfunction_one
void testfunction_one(A a)
{
    // CHECK: load {{.*}} !invariant.group
    // CHECK: [[RET1:%[0-9]]] = load {{.*}} !invariant.load
    // CHECK: call {{.*}} [[RET1]](
    // CHECK-NEXT: call {{.*}} [[RET1]](
    a.foo();
    a.foo();
}

// CHECK-LABEL: define{{.*}} @{{.*}}testfunction_forloop
int testfunction_forloop()
{
    // Because the final type of `a` is known, the call can be fully devirtualized for all iterations of the loop.
    // This should result in a fully optimized function.
    // CHECK: ret i32 100

    auto a = new A();
    int sum = 0;
    for (int i = 0; i < 100; i++)
    {
        sum += a.one();
    }
    return sum;
}

// CHECK-LABEL: define{{.*}} @{{.*}}testfunction_two
void testfunction_two(A a, A b)
{
    // CHECK: load {{.*}} !invariant.group
    // CHECK: [[RET1:%[0-9]]] = load {{.*}} !invariant.load
    // CHECK: call {{.*}} [[RET1]](
    a.foo();
    a = b;
    // CHECK: load {{.*}} !invariant.group
    // CHECK: [[RET2:%[0-9]]] = load {{.*}} !invariant.load
    // CHECK: call {{.*}} [[RET2]](
    a.foo();
}

// This opaque function may change the vptr.
void clobber_by_ref(ref A a);

// CHECK-LABEL: define{{.*}} @{{.*}}testfunction_three
void testfunction_three(A a)
{
    // CHECK: load {{.*}} !invariant.group
    // CHECK: [[RET3_1:%[0-9]]] = load {{.*}} !invariant.load
    // CHECK: call {{.*}} [[RET3_1]](
    a.foo();
    clobber_by_ref(a);
    // CHECK: load {{.*}}
    // CHECK: load {{.*}} !invariant.group
    // CHECK: [[RET3_2:%[0-9]]] = load {{.*}} !invariant.load
    // CHECK: call {{.*}} [[RET3_2]](
    a.foo();
}

// Because `a` is passed by value, this opaque function cannot change the vptr without
// resulting in UB if `a` is used afterwards.
void may_delete_destroy(A a);

// CHECK-LABEL: define{{.*}} @{{.*}}testfunction_four
void testfunction_four(A a)
{
    // CHECK: load {{.*}} !invariant.group
    // CHECK: [[RET4_1:%[0-9]]] = load {{.*}} !invariant.load
    // CHECK: call {{.*}} [[RET4_1]](
    // CHECK-NEXT: call {{.*}}may_delete_destroy
    // CHECK-NEXT: call {{.*}} [[RET4_1]](
    a.foo();
    may_delete_destroy(a);
    a.foo();
}

class B : A
{
    override void foo();
}

struct AB
{
    union
    {
        A a;
        B b;
    }
}

void clobber(ref AB);

// CHECK-LABEL: define{{.*}} @{{.*}}testfunction_five
void testfunction_five(AB ab)
{
    // CHECK: load {{.*}} !invariant.group
    // CHECK: [[RET5_1:%[0-9]+]] = load {{.*}} !invariant.load
    // CHECK: call {{.*}} [[RET5_1]](
    // CHECK-NEXT: call {{.*}}7clobber
    // CHECK: load {{.*}} !invariant.group
    // CHECK: [[RET5_2:%[0-9]+]] = load {{.*}} !invariant.load
    // CHECK: call {{.*}} [[RET5_2]](
    ab.a.foo();
    clobber(ab);
    ab.b.foo();
}

// Test that if(a.vptr) is still working properly after may_delete_destroy that could do a delete.
// Note: this is underspecced, we are perhaps needlessly extra careful here.
// Note that __vptr is not pointing directly to the vtable.
// CHECK-LABEL: define{{.*}} @{{.*}}testfunction_vpointer
void testfunction_vpointer(A a)
{
    // CHECK: load {{.*}} !invariant.group
    // CHECK: [[RETvpointer_foo:%[0-9]+]] = load {{.*}} !invariant.load
    // CHECK: call {{.*}} [[RETvpointer_foo]](
    a.foo();

    // CHECK: [[RETvpointer_load1:%[0-9]+]] = load i8**
    // CHECK: [[RETvpointer_1:%[0-9]+]] = icmp eq i8** [[RETvpointer_load1]], null
    // CHECK: br i1 [[RETvpointer_1]]
    if (a.__vptr)
    {
        // CHECK: call {{.*}}may_delete_destroy
        may_delete_destroy(a);

        // This 2nd check of the __vptr should not be eliminated.
        // CHECK: [[RETvpointer_load2:%[0-9]+]] = load i8**
        // CHECK: [[RETvpointer_2:%[0-9]+]] = icmp eq i8** [[RETvpointer_load2]], null
        // CHECK: br i1 [[RETvpointer_2]]
        if (a.__vptr)
        {
            // We could still use the previously loaded pointer to A::foo, but the
            // optimizer is not strong enough yet.
            // disabledCHECK: call {{.*}} [[RETvpointer_foo]](
            a.foo();
        }
    }
}

// Test calls through an interface
interface I
{
    void foo();
}

interface I2
{
    void ggg();
}

class IA : I, I2
{
    override void foo();
    override void ggg();
}

class IB : IA
{
    override void foo();
}

// CHECK-LABEL: define{{.*}} @{{.*}}testinterface_zero
void testinterface_zero(IA a)
{
    // CHECK: load {{.*}} !invariant.group
    // CHECK: [[INTFC0:%[0-9]]] = load {{.*}} !invariant.load
    // CHECK: call {{.*}} [[INTFC0]](
    a.foo();
    // CHECK-NEXT: call {{.*}} [[INTFC0]](
    a.foo();
    // CHECK: [[INTFC0_2:%[0-9]]] = load {{.*}} !invariant.load
    // CHECK: call {{.*}} [[INTFC0_2]](
    a.ggg();
}

// CHECK-LABEL: define{{.*}} @{{.*}}testinterface_one
void testinterface_one(I a)
{
    // CHECK: load {{.*}} !invariant.group
    // CHECK: [[INTFC1:%[0-9]]] = load {{.*}} !invariant.load
    // CHECK: call {{.*}} [[INTFC1]](
    // CHECK-NEXT: call {{.*}} [[INTFC1]](
    a.foo();
    a.foo();
}

// CHECK-LABEL: define{{.*}} @{{.*}}testinterface_two
void testinterface_two(I a, I b)
{
    // CHECK: load {{.*}} !invariant.group
    // CHECK: [[INTFC1:%[0-9]]] = load {{.*}} !invariant.load
    // CHECK: call {{.*}} [[INTFC1]](
    a.foo();
    a = b;
    // CHECK: load {{.*}} !invariant.group
    // CHECK: [[INTFC2:%[0-9]]] = load {{.*}} !invariant.load
    // CHECK: call {{.*}} [[INTFC2]](
    a.foo();
}

// This opaque function may change the vptr.
void clobber_by_ref(ref I a);

// CHECK-LABEL: define{{.*}} @{{.*}}testinterface_three
void testinterface_three(I a)
{
    // CHECK: load {{.*}} !invariant.group
    // CHECK: [[INTFC3_1:%[0-9]]] = load {{.*}} !invariant.load
    // CHECK: call {{.*}} [[INTFC3_1]](
    a.foo();
    clobber_by_ref(a);
    // CHECK: load {{.*}}
    // CHECK: load {{.*}} !invariant.group
    // CHECK: [[INTFC3_2:%[0-9]]] = load {{.*}} !invariant.load
    // CHECK: call {{.*}} [[INTFC3_2]](
    a.foo();
}

// Because `a` is passed by value, this opaque function cannot change the vptr without
// resulting in UB if `a` is used afterwards.
void may_delete_destroy(I2 a);

// CHECK-LABEL: define{{.*}} @{{.*}}testinterface_four
void testinterface_four(I2 a)
{
    // CHECK: load {{.*}} !invariant.group
    // CHECK: [[INTFC4_1:%[0-9]]] = load {{.*}} !invariant.load
    // CHECK: call {{.*}} [[INTFC4_1]](
    a.ggg();
    // CHECK-NEXT: call {{.*}}may_delete_destroy
    may_delete_destroy(a);
    // CHECK-NEXT: call {{.*}} [[INTFC4_1]](
    a.ggg();
}
