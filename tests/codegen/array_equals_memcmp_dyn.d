// Tests that dynamic array (in)equality is optimized to a memcmp call when valid.

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s --check-prefix=LLVM < %t.ll
// RUN: %ldc -c -output-s -O3 -of=%t.s  %s && FileCheck %s --check-prefix=ASM  < %t.s
// RUN: %ldc -O0 -run %s
// RUN: %ldc -O3 -run %s

module mod;

// LLVM-LABEL: define{{.*}} @{{.*}}static_dynamic
// ASM-LABEL: static_dynamic{{.*}}:
bool static_dynamic(const bool[4] a, bool[] b)
{
    // LLVM: call i32 @memcmp(

    // Also test that LLVM recognizes and optimizes-out the call to memcmp for 4 byte arrays:
    // ASM-NOT: {{(mem|b)cmp}}
    return a == b;
}

// LLVM-LABEL: define{{.*}} @{{.*}}inv_dynamic_dynamic
// ASM-LABEL: inv_dynamic_dynamic{{.*}}:
bool inv_dynamic_dynamic(bool[] a, const bool[] b)
{
    // The front-end turns this into a call to druntime template function `object.__equals!(const(bool), const(bool)).__equals(const(bool)[], const(bool)[])`
    // After optimization (inlining), it should boil down to a length check and a call to memcmp.
    // ASM: {{(mem|b)cmp}}
    return a != b;
}

// LLVM-LABEL: define{{.*}} @{{.*}}_D4core8internal5array8equality__T8__equals
// ASM-LABEL: _D4core8internal5array8equality__T8__equals{{.*}}:

// LLVM-LABEL: define{{.*}} @_Dmain
// ASM-LABEL: _Dmain:
void main()
{
    assert( static_dynamic([true, false, true, false], [true, false, true, false]));
    assert(!static_dynamic([true, false, true, false], [true, false, true, true]));
    assert(!static_dynamic([true, false, true, false], [true, false, true, false, true]));
    assert(!static_dynamic([true, false, true, false], [true, false, true]));

    assert(!inv_dynamic_dynamic([true, false, true, false], [true, false, true, false]));
    assert( inv_dynamic_dynamic([true, false, true, false], [true, false, true, true]));
    assert( inv_dynamic_dynamic([true, false], [true]));
    assert( inv_dynamic_dynamic([true, false, true, false], [true, false, true]));

    // Make sure that comparing zero-length arrays with ptr=null is allowed.
    bool* ptr = null;
    assert(!inv_dynamic_dynamic(ptr[0..0], ptr[0..0]));
}
