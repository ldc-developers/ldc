// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// Make sure the LL struct is packed due to an unnatural overall alignment of 1.
// CHECK-DAG: %gh2346.UnalignedUInt = type <{ i32 }>
struct UnalignedUInt {
    align(1) uint a;
}
static assert(UnalignedUInt.alignof == 1);
static assert(UnalignedUInt.sizeof == 4);

// Then there's no need to pack naturally aligned containers.
// CHECK-DAG: %gh2346.Container = type { i8, %gh2346.UnalignedUInt }
struct Container {
    ubyte one;
    UnalignedUInt two;
}
static assert(Container.alignof == 1);
static assert(Container.sizeof == 5);
static assert(Container.two.offsetof == 1);

// CHECK-DAG: %gh2346.UnalignedUInt2 = type <{ i32 }>
struct UnalignedUInt2 {
    align(2) uint a;
}
static assert(UnalignedUInt2.alignof == 2);
static assert(UnalignedUInt2.sizeof == 4);

// CHECK-DAG: %gh2346.Container2 = type { i8, [1 x i8], %gh2346.UnalignedUInt2 }
struct Container2 {
    ubyte one;
    UnalignedUInt2 two;
}
static assert(Container2.alignof == 2);
static assert(Container2.sizeof == 6);
static assert(Container2.two.offsetof == 2);

// CHECK-DAG: %gh2346.PackedContainer2 = type <{ i8, %gh2346.UnalignedUInt2 }>
struct PackedContainer2 {
    ubyte one;
    align(1) UnalignedUInt2 two;
}
static assert(PackedContainer2.alignof == 1);
static assert(PackedContainer2.sizeof == 5);
static assert(PackedContainer2.two.offsetof == 1);

// CHECK-DAG: %gh2346.WeirdContainer = type { i8, [1 x i8], %gh2346.UnalignedUInt, [2 x i8] }
align(4) struct WeirdContainer {
    ubyte one;
    align(2) UnalignedUInt two;
}
static assert(WeirdContainer.alignof == 4);
static assert(WeirdContainer.sizeof == 8);
static assert(WeirdContainer.two.offsetof == 2);

// CHECK-DAG: %gh2346.ExplicitlyUnalignedUInt2 = type <{ i32 }>
align(2) struct ExplicitlyUnalignedUInt2 {
    uint a;
}
static assert(ExplicitlyUnalignedUInt2.alignof == 2);
static assert(ExplicitlyUnalignedUInt2.sizeof == 4);

// CHECK-DAG: %gh2346.AnotherContainer = type { i8, [1 x i8], %gh2346.ExplicitlyUnalignedUInt2 }
struct AnotherContainer {
    ubyte one;
    ExplicitlyUnalignedUInt2 two;
}
static assert(AnotherContainer.alignof == 2);
static assert(AnotherContainer.sizeof == 6);
static assert(AnotherContainer.two.offsetof == 2);

// reference all types
void foo()
{
    Container a;
    Container2 b;
    PackedContainer2 c;
    WeirdContainer d;
    AnotherContainer e;
}
