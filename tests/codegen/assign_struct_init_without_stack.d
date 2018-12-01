// Tests that even in non-optimized builds, member = member.init immediately does a memcpy,
// instead of going through a temporary stack allocated variable.
// Exception: when member has an opAssign with auto ref parameter a local temporary
// has to be passed because `.init` is an rvalue,

// RUN: %ldc -output-ll %s -of=%t.ll && FileCheck %s < %t.ll

void opaque();
FloatStruct opaque(FloatStruct);

struct FloatStruct {
    float[10] data;
}

FloatStruct globalStruct;

// CHECK-LABEL: define{{.*}} @{{.*}}_D32assign_struct_init_without_stack3fooFZv
void foo() {
    globalStruct = FloatStruct.init;
    // There should be only one memcpy.
    // CHECK: call void @llvm.memcpy
    // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D32assign_struct_init_without_stack3hhhFZv
void hhh() {
    globalStruct = FloatStruct([1, 0, 0, 0, 0, 0, 0, 0, 0, 42]);
    // Future work: test optimized codegen (at -O0)
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D32assign_struct_init_without_stack3gggFZv
void ggg() {
    globalStruct = opaque(globalStruct);
    // There should be one memcpy from a temporary (sret return).
    // CHECK: alloca %assign_struct_init_without_stack.FloatStruct
    // CHECK: call
    // CHECK: call void @llvm.memcpy
    // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D32assign_struct_init_without_stack5arrayFZv
void array() {
    int[5] arr = [0,1,2,3,4];
    // Future work: test optimized codegen (at -O0)
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D32assign_struct_init_without_stack6array2FKG3iZv
void array2(ref int[3] a) {
    a = [4, a[0], 6];
    // There should be a temporary!
    // CHECK: alloca [3 x i32]
    // CHECK: call void @llvm.memcpy
    // CHECK-NEXT: ret void
}

struct OpAssignStruct {
    float[10] data;
    ubyte a;

    ref OpAssignStruct opAssign(R)(auto ref R rhs) {
        return this;
    }
}
OpAssignStruct globalOpAssignStruct;
OpAssignStruct globalOpAssignStruct2;

// CHECK-LABEL: define{{.*}} @{{.*}}_D32assign_struct_init_without_stack16tupleassignByValFZv
void tupleassignByVal()
{
    globalOpAssignStruct = OpAssignStruct.init;
    // There should be one memcpy to a temporary.
    // CHECK: alloca %assign_struct_init_without_stack.OpAssignStruct
    // CHECK: call void @llvm.memcpy
    // CHECK-NOT: memcpy
    // CHECK: call{{.*}} %assign_struct_init_without_stack.OpAssignStruct* @{{.*}}_D32assign_struct_init_without_stack14OpAssignStruct__T8opAssignTSQCmQBhZQsMFNaNbNcNiNjNfQyZQBb
    // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D32assign_struct_init_without_stack16tupleassignByRefFZv
void tupleassignByRef()
{
    globalOpAssignStruct = globalOpAssignStruct2;
    // There should not be a memcpy.
    // CHECK-NOT: memcpy
    // CHECK: call{{.*}} %assign_struct_init_without_stack.OpAssignStruct* @{{.*}}_D32assign_struct_init_without_stack14OpAssignStruct__T8opAssignTSQCmQBhZQsMFNaNbNcNiNjNfKQzZQBc
    // CHECK-NEXT: ret void
}

struct DtorStruct {
    float[10] data;
    ~this() { opaque(); }
}
struct CtorStruct {
    float[10] data;
    this(int i) { opaque(); }
}
DtorStruct dtorStruct;
CtorStruct ctorStruct;

// CHECK-LABEL: define{{.*}} @{{.*}}_D32assign_struct_init_without_stack4ctorFZv
void ctor() {
    ctorStruct = ctorStruct.init;
    // There is no dtor, so can be optimized to only a memcpy.
    // CHECK-NEXT: call void @llvm.memcpy
    // CHECK-NEXT: ret void
}
// CHECK-LABEL: define{{.*}} @{{.*}}_D32assign_struct_init_without_stack4dtorFZv
void dtor() {
    dtorStruct = dtorStruct.init;
    // There should be a temporary and a call to opAssign
    // CHECK: alloca %assign_struct_init_without_stack.DtorStruct
    // CHECK: call void @llvm.memcpy{{.*}}_D32assign_struct_init_without_stack10DtorStruct6__initZ
    // CHECK-NEXT: call {{.*}}_D32assign_struct_init_without_stack10DtorStruct8opAssignMFNcNjSQCkQBfZQi
    // CHECK-NEXT: ret void
}
