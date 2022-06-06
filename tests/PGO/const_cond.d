// Test that generation of PGO counters is the same as with
// the no-elision version.

// RUN: %ldc -c -output-ll -fprofile-instr-generate -of=%t.ll %s && FileCheck %s < %t.ll


extern(C):

// Test that we have exactly 2 counters for `foo` (one for the function, one for the `if`)
// and only one counter for `bar` (bar has always false if, should be elided).
// CHECK: @__{{(llvm_profile_counters|profc)}}_foo ={{.*}} [2 x i64] zeroinitializer
// We want 2 counters here too, no matter that the second one (for the `if`) is not referenced.
// It may be queried by the user.
// CHECK: @__{{(llvm_profile_counters|profc)}}_bar ={{.*}} [2 x i64] zeroinitializer


// CHECK-LABEL: @foo()
// - Counters appear in triads:
// 1) load counter
// 2) increment it
// 3) store back
// The first 2 contain a `pgocount*` named register.
void foo() {
    // CHECK-NOT: pgocount
    // CHECK: pgocount
    // CHECK: pgocount
    // CHECK: pgocount
    // CHECK: pgocount
    // CHECK-NOT: pgocount
    if (true)
    {
        int a;
    }
}

// CHECK-LABEL: @bar
void bar() {
    // CHECK-NOT: pgocount
    // CHECK: pgocount
    // CHECK: pgocount
    // CHECK-NOT: pgocount
    if (false)
    {
        int a;
    }
}
