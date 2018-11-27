// Tests that even in non-optimized builds, member = member.init immediately does a memcpy,
// instead of going through a temporary stack allocated variable.
// Exception: when member has an opAssign with auto ref parameter, then the .init symbol
// apparently counts as an rvalue so a local temporary has to be passed.

// RUN: %ldc -output-ll %s -of=%t.ll && FileCheck %s < %t.ll

struct FloatStruct {
    float[10] data;
}
FloatStruct opaque(FloatStruct);

FloatStruct globalStruct;

// CHECK-LABEL: _D32assign_struct_init_without_stack3fooFZv
void foo() {
    globalStruct = FloatStruct.init;
    // There should be only one memcpy.
    // CHECK: memcpy
    // CHECK-NEXT: ret void
}

// CHECK-LABEL: _D32assign_struct_init_without_stack3hhhFZv
void hhh() {
    globalStruct = FloatStruct([1, 0, 0, 0, 0, 0, 0, 0, 0, 42]);
    // There should be only one memcpy.
    // CHECK: memcpy
    // CHECK-NEXT: ret void
}

// CHECK-LABEL: _D32assign_struct_init_without_stack3gggFZv
void ggg() {
    globalStruct = opaque(globalStruct);
    // There should be one memcpy from a temporary (sret return).
    // CHECK: alloca %assign_struct_init_without_stack.FloatStruct
    // CHECK: call
    // CHECK: memcpy
    // CHECK-NEXT: ret void
}

// CHECK-LABEL: _D32assign_struct_init_without_stack5arrayFZv
void array() {
    int[5] arr = [0,1,2,3,4];
    // There should be one memcpy.
    // CHECK: memcpy
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

// CHECK-LABEL: _D32assign_struct_init_without_stack16tupleassignByValFZv
void tupleassignByVal()
{
    globalOpAssignStruct = OpAssignStruct.init;
    // There should be one memcpy to a temporary.
    // CHECK: alloca %assign_struct_init_without_stack.OpAssignStruct
    // CHECK: memcpy
    // CHECK-NOT: memcpy
    // CHECK: call %assign_struct_init_without_stack.OpAssignStruct* @_D32assign_struct_init_without_stack14OpAssignStruct__T8opAssignTSQCmQBhZQsMFNaNbNcNiNjNfQyZQBb
    // CHECK-NEXT: ret void
}

// CHECK-LABEL: _D32assign_struct_init_without_stack16tupleassignByRefFZv
void tupleassignByRef()
{
    globalOpAssignStruct = globalOpAssignStruct2;
    // There should not be a memcpy.
    // CHECK-NOT: memcpy
    // CHECK: call %assign_struct_init_without_stack.OpAssignStruct* @_D32assign_struct_init_without_stack14OpAssignStruct__T8opAssignTSQCmQBhZQsMFNaNbNcNiNjNfKQzZQBc
    // CHECK-NEXT: ret void
}
