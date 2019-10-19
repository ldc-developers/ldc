// Tests that static array (in)equality is optimized to a memcmp call when valid.
// More importantly: test that memcmp is _not_ used when it is not valid.

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s --check-prefix=LLVM < %t.ll
// RUN: %ldc -O3 -c -output-s  -of=%t.s  %s && FileCheck %s --check-prefix=ASM  < %t.s
// RUN: %ldc -O3 -run %s

module mod;

struct ThreeBytes
{
    byte a;
    byte b;
    byte c;
}

align(4) struct ThreeBytesAligned
{
    byte a;
    byte b;
    byte c;
}

struct Packed
{
    byte a;
    byte b;
    byte c;
    byte d;
}

struct PackedPacked
{
    Packed a;
    Packed b;
}

struct WithPadding
{
    int b;
    byte a;
}

// LLVM-LABEL: define{{.*}} @{{.*}}two_uints
bool two_uints(ref uint[2] a, const ref uint[2] b)
{
    // LLVM: call i32 @memcmp({{.*}}, {{.*}}, i{{32|64}} 8)
    return a == b;
}

// LLVM-LABEL: define{{.*}} @{{.*}}unequal_two_uints
bool unequal_two_uints(ref uint[2] a, uint[2] b)
{
    // LLVM: call i32 @memcmp({{.*}}, {{.*}}, i{{32|64}} 8)
    return a != b;
}

// LLVM-LABEL: define{{.*}} @{{.*}}two_floats
bool two_floats(float[2] a, float[2] b)
{
    // LLVM-NOT: memcmp
    return a == b;
}

// LLVM-LABEL: define{{.*}} @{{.*}}four_bools
// ASM-LABEL: four_bools{{.*}}:
bool four_bools(bool[4] a, bool[4] b)
{
    // LLVM: call i32 @memcmp({{.*}}, {{.*}}, i{{32|64}} 4)

    // Make sure that LLVM recognizes and optimizes-out the call to memcmp for 4 byte arrays:
    // ASM-NOT: {{(mem|b)cmp}}
    return a == b;
}

// LLVM-LABEL: define{{.*}} @{{.*}}array_of_array
// ASM-LABEL: array_of_array{{.*}}:
bool array_of_array(byte[3][3] a, const byte[3][3] b)
{
    // LLVM: call i32 @memcmp({{.*}}, {{.*}}, i{{32|64}} 9)
    return a == b;
}

// LLVM-LABEL: define{{.*}} @{{.*}}int3_short3
bool int3_short3(int[3] a, short[3] b)
{
    // LLVM-NOT: memcmp
    return a == b;
    // LLVM-LABEL: ret i1
}

// LLVM-LABEL: define{{.*}} @{{.*}}pointer3
bool pointer3(int*[3] a, int*[3] b)
{
    // LLVM: call i32 @memcmp({{.*}}, {{.*}}, i{{32|64}} {{12|24}})
    return a == b;
}

// LLVM-LABEL: define{{.*}} @{{.*}}enum3
enum E : char { a, b, c, d, e, f };
bool enum3(E[3] a, E[3] b)
{
    // LLVM: call i32 @memcmp({{.*}}, {{.*}}, i{{32|64}} 3)
    return a == b;
}

class K {}
// LLVM-LABEL: define{{.*}} @{{.*}}klass2
bool klass2(K[2] a, K[2] b)
{
    // LLVM-NOT: memcmp
    return a == b;
    // LLVM-LABEL: ret i1
}

void main()
{
    uint[2] a = [1, 2];
    uint[2] b = [1, 2];
    uint[2] c = [2, 1];
    assert(two_uints(a, a));
    assert(two_uints(a, b));
    assert(!two_uints(a, c));
    assert(!unequal_two_uints(a, b));
    assert(unequal_two_uints(a, c));

    assert( two_floats([1.0f, 2.0f], [1.0f, 2.0f]));
    assert(!two_floats([1.0f, 2.0f], [2.0f, 1.0f]));

    assert( four_bools([true, false, true, false], [true, false, true, false]));
    assert(!four_bools([true, false, true, false], [true, false, true, true]));

    assert( array_of_array([[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]));
    assert(!array_of_array([[1,2,3],[4,5,6],[7,8,9]],[[6,6,6],[4,5,6],[7,8,9]]));

    assert( int3_short3([1, 2, 3], [1, 2, 3]));
    assert(!int3_short3([1, 2, 3], [3, 2, 3]));

    int aaa = 666;
    int bbb = 333;
    assert( pointer3([&aaa, &bbb, &aaa], [&aaa, &bbb, &aaa]));
    assert(!pointer3([&aaa, &bbb, &aaa], [&bbb, &bbb, &aaa]));

    assert( enum3([E.a, E.e, E.b], [E.a, E.e, E.b]));
    assert(!enum3([E.a, E.e, E.b], [E.a, E.e, E.f]));
}
