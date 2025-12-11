// It should compile fine when not referencing the colliding external global:
// RUN: %ldc -c %s -d-version=DontReference

// But fail if referenced:
// RUN: not %ldc -c %s -verrors-context=false 2>&1 | FileCheck %s

extern(C) extern int myGlobal;

version (DontReference) {} else
{
    int dummyRef() { return myGlobal; }
}

// CHECK:      global_var_collision.d([[@LINE+4]]): Error: Global variable type does not match previous declaration with same mangled name: `myGlobal`
// CHECK-NEXT: Previous IR type: i32, mutable, thread-local
// CHECK-NEXT: New IR type:      i64, const, non-thread-local
pragma(mangle, myGlobal.mangleof)
extern(C) __gshared const long myGlobal2 = 123;
