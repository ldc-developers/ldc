// Tests that dynamic array (in)equality is optimized to a memcmp call when valid.

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s --check-prefix=LLVM < %t.ll
// RUN: %ldc -O3 -run %s

module mod;

// LLVM-LABEL: define{{.*}} @{{.*}}static_dynamic
bool static_dynamic(bool[4] a, bool[] b)
{
    // LLVM: call i32 @memcmp(
    return a == b;
}

// LLVM-LABEL: define{{.*}} @{{.*}}inv_dynamic_dynamic
bool inv_dynamic_dynamic(bool[] a, bool[] b)
{
    // LLVM: call i32 @memcmp(
    return a != b;
}

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
}
