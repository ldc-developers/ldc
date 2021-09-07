// Tests LL types and constant initializers of init symbols and globals of
// structs with and without overlapping (union) fields.

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -run %s

struct S
{
    char c;              // default initializer: 0xff
    uint ui;
    bool[2] bools;       // make sure the 2 bools are extended to 2 bytes
    bool b = true;       // scalar bool too
    char[2][1] multidim; // multidimensional init based on a single 0xff char
}
// CHECK-DAG: %union.S                              = type { i8, [3 x i8], i32, [2 x i8], i8, [1 x [2 x i8]], [3 x i8] }
// CHECK-DAG: @{{.*}}_D5union1S6__initZ{{\"?}}      = constant %union.S { i8 -1, [3 x i8] zeroinitializer, i32 0, [2 x i8] zeroinitializer, i8 1, [1 x [2 x i8]] {{\[}}[2 x i8] c"\FF\FF"], [3 x i8] zeroinitializer }

// CHECK-DAG: @{{.*}}_D5union8defaultSSQq1S{{\"?}}  = global   %union.S { i8 -1, [3 x i8] zeroinitializer, i32 0, [2 x i8] zeroinitializer, i8 1, [1 x [2 x i8]] {{\[}}[2 x i8] c"\FF\FF"], [3 x i8] zeroinitializer }
__gshared S defaultS;

// CHECK-DAG: @{{.*}}_D5union9explicitSSQr1S{{\"?}} = global   %union.S { i8 3, [3 x i8] zeroinitializer, i32 56, [2 x i8] c"\00\01", i8 0, [1 x [2 x i8]] {{\[}}[2 x i8] c"\FF\FF"], [3 x i8] zeroinitializer }
__gshared S explicitS = { 3, 56, [false, true], false /* implicit multidim */ };



struct SWithUnion
{
    char c;
    S nested;

    union
    {
        struct { ubyte ub = 6; ushort us = 33;                align(8) ulong ul = 666; }
        struct { uint ui1;                     uint ui2 = 84; ulong ul_dummy;          ulong last = 123; }
    }
}
// CHECK-DAG: %union.SWithUnion                                                      = type { i8, [3 x i8], %union.S, [4 x i8], i8, [1 x i8], i16, i32, i64, i64 }
// CHECK-DAG: @{{.*}}_D5union10SWithUnion6__initZ{{\"?}}                             = constant %union.SWithUnion { i8 -1, [3 x i8] zeroinitializer, %union.S { i8 -1, [3 x i8] zeroinitializer, i32 0, [2 x i8] zeroinitializer, i8 1, [1 x [2 x i8]] {{\[}}[2 x i8] c"\FF\FF"], [3 x i8] zeroinitializer }, [4 x i8] zeroinitializer, i8 6, [1 x i8] zeroinitializer, i16 33, i32 84, i64 666, i64 123 }

// CHECK-DAG: @{{.*}}_D5union17defaultSWithUnionSQBa10SWithUnion{{\"?}}              = global   %union.SWithUnion { i8 -1, [3 x i8] zeroinitializer, %union.S { i8 -1, [3 x i8] zeroinitializer, i32 0, [2 x i8] zeroinitializer, i8 1, [1 x [2 x i8]] {{\[}}[2 x i8] c"\FF\FF"], [3 x i8] zeroinitializer }, [4 x i8] zeroinitializer, i8 6, [1 x i8] zeroinitializer, i16 33, i32 84, i64 666, i64 123 }
__gshared SWithUnion defaultSWithUnion;

// CHECK-DAG: @{{.*}}_D5union28explicitCompatibleSWithUnionSQBl10SWithUnion{{\"?}}   = global   %union.SWithUnion { i8 -1, [3 x i8] zeroinitializer, %union.S { i8 -1, [3 x i8] zeroinitializer, i32 0, [2 x i8] zeroinitializer, i8 1, [1 x [2 x i8]] {{\[}}[2 x i8] c"\FF\FF"], [3 x i8] zeroinitializer }, [4 x i8] zeroinitializer, i8 6, [1 x i8] zeroinitializer, i16 33, i32 84, i64 53, i64 123 }
__gshared SWithUnion explicitCompatibleSWithUnion = { ul_dummy: 53 }; // ul_dummy is an alias for dominant ul

// If a dominated union field is initialized and it isn't an alias for a dominant field,
// the regular LL type cannot be used, and an anonymous one is used instead.
// CHECK-DAG: @{{.*}}_D5union30explicitIncompatibleSWithUnionSQBn10SWithUnion{{\"?}} = global   { i8, [3 x i8], %union.S, [4 x i8], i32, i32, i64, i64 } { i8 -1, [3 x i8] zeroinitializer, %union.S { i8 -1, [3 x i8] zeroinitializer, i32 0, [2 x i8] zeroinitializer, i8 1, [1 x [2 x i8]] {{\[}}[2 x i8] c"\FF\FF"], [3 x i8] zeroinitializer }, [4 x i8] zeroinitializer, i32 23, i32 84, i64 666, i64 123 }
__gshared SWithUnion explicitIncompatibleSWithUnion = { ui1: 23 }; // // ui1 dominated by ub and us



struct Quat
{
    static struct Vec { int x; }

    union
    {
        Vec v;
        struct { float x; }
    }

    static Quat identity()
    {
        Quat q;
        q.x = 1.0f;
        return q;
    }
}

// T.init may feature explicit initializers for dominated members in nested unions (GitHub issue #2108).
// In that case, the init constant has an anonymous LL type as well.
// CHECK-DAG: @{{.*}}_D5union33QuatContainerWithIncompatibleInit6__initZ{{\"?}} = constant { { float } } { { float } { float 1.000000e+00 } }
struct QuatContainerWithIncompatibleInit
{
    Quat q = Quat.identity;
}



void main()
{
    // test dynamic literals too

    {
        SWithUnion s = { 'y' };
        assert(s.c == 'y');
        assert(s.nested == S.init);
        assert(s.ub == 6);
        assert(s.us == 33);
        assert(s.ui2 == 84);
        assert(s.ul == 666);
        assert(s.last == 123);
    }

    {
        SWithUnion s = { ul_dummy: 53 };
        assert(s.c == char.init);
        assert(s.nested == S.init);
        assert(s.ub == 6);
        assert(s.us == 33);
        assert(s.ui2 == 84);
        assert(s.ul_dummy == 53);
        assert(s.last == 123);
    }

    {
        SWithUnion s = { ui1: 23 };
        assert(s.c == char.init);
        assert(s.nested == S.init);
        assert(s.ui1 == 23);
        assert(s.ui2 == 84);
        assert(s.ul == 666);
        assert(s.last == 123);
    }

    {
        QuatContainerWithIncompatibleInit c;
        assert(c.q.x == 1.0f);
    }
}
