// Tests LL types and constant initializers of init symbols and globals of
// structs with and without overlapping (union) fields.

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

struct S
{
    char c;              // default initializer: 0xff
    uint ui;
    bool[2] bools;       // make sure the 2 bools are extended to 2 bytes
    char[2][4] multidim; // multidimensional init based on a single 0xff char
}
// CHECK-DAG: %union.S                     = type { i8, [3 x i8], i32, [2 x i8], [4 x [2 x i8]], [2 x i8] }
// CHECK-DAG: @_D5union1S6__initZ          = constant %union.S { i8 -1, [3 x i8] zeroinitializer, i32 0, [2 x i8] zeroinitializer, [4 x [2 x i8]] {{\[}}[2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF"], [2 x i8] zeroinitializer }

// CHECK-DAG: @_D5union8defaultSS5union1S  = global   %union.S { i8 -1, [3 x i8] zeroinitializer, i32 0, [2 x i8] zeroinitializer, [4 x [2 x i8]] {{\[}}[2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF"], [2 x i8] zeroinitializer }
__gshared S defaultS;

// CHECK-DAG: @_D5union9explicitSS5union1S = global   %union.S { i8 3, [3 x i8] zeroinitializer, i32 56, [2 x i8] c"\00\01", [4 x [2 x i8]] {{\[}}[2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF"], [2 x i8] zeroinitializer }
__gshared S explicitS = S(3, 56, [false, true] /* implicit multidim */);


struct SWithUnion
{
    char c;
    S nested;

    union
    {
        struct { ubyte ub = 6; ushort us = 33; ulong ul_dummy = void; }
        struct { uint ui1; uint ui2 = 84; ulong ul = 666; }
        ubyte ub_direct;
    }

    this(ubyte ub)
    {
        this.ub_direct = ub; // ub_direct is an alias for dominant ub
    }

    this(uint ui1, uint ui2)
    {
        this.ui1 = ui1; // dominated by ub and us
        this.ui2 = ui2;
    }
}
// CHECK-DAG: %union.SWithUnion                                            = type { i8, [3 x i8], %union.S, i8, [1 x i8], i16, i32, i64 }
// CHECK-DAG: @_D5union10SWithUnion6__initZ                                = constant %union.SWithUnion { i8 -1, [3 x i8] zeroinitializer, %union.S { i8 -1, [3 x i8] zeroinitializer, i32 0, [2 x i8] zeroinitializer, [4 x [2 x i8]] {{\[}}[2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF"], [2 x i8] zeroinitializer }, i8 6, [1 x i8] zeroinitializer, i16 33, i32 84, i64 666 }

// CHECK-DAG: @_D5union17defaultSWithUnionS5union10SWithUnion              = global   %union.SWithUnion { i8 -1, [3 x i8] zeroinitializer, %union.S { i8 -1, [3 x i8] zeroinitializer, i32 0, [2 x i8] zeroinitializer, [4 x [2 x i8]] {{\[}}[2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF"], [2 x i8] zeroinitializer }, i8 6, [1 x i8] zeroinitializer, i16 33, i32 84, i64 666 }
__gshared SWithUnion defaultSWithUnion;

// CHECK-DAG: @_D5union28explicitCompatibleSWithUnionS5union10SWithUnion   = global   %union.SWithUnion { i8 -1, [3 x i8] zeroinitializer, %union.S { i8 -1, [3 x i8] zeroinitializer, i32 0, [2 x i8] zeroinitializer, [4 x [2 x i8]] {{\[}}[2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF"], [2 x i8] zeroinitializer }, i8 53, [1 x i8] zeroinitializer, i16 33, i32 84, i64 666 }
__gshared SWithUnion explicitCompatibleSWithUnion = SWithUnion(53);

// When using the 2nd constructor, a dominated union field (ui1) is initialized.
// The regular LL type can thus not be used, an anonymous one is used instead.
// CHECK-DAG: @_D5union30explicitIncompatibleSWithUnionS5union10SWithUnion = global   { i8, [3 x i8], %union.S, i32, i32, i64 } { i8 -1, [3 x i8] zeroinitializer, %union.S { i8 -1, [3 x i8] zeroinitializer, i32 0, [2 x i8] zeroinitializer, [4 x [2 x i8]] {{\[}}[2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF", [2 x i8] c"\FF\FF"], [2 x i8] zeroinitializer }, i32 23, i32 256, i64 666 }
__gshared SWithUnion explicitIncompatibleSWithUnion = SWithUnion(23, 256);
