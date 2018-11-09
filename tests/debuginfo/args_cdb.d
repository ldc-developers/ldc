
// REQUIRES: Windows
// REQUIRES: cdb
// RUN: %ldc -g -of=%t.exe %s
// RUN: sed -e "/^\\/\\/ CDB:/!d" -e "s,// CDB:,," %s \
// RUN:    | %cdb -snul -lines -y . %t.exe >%t.out
// RUN: FileCheck %s -check-prefix=CHECK -check-prefix=%arch < %t.out

module args_cdb;
import core.simd;

// CDB: ld /f args_cdb*
// enable case sensitive symbol lookup
// CDB: .symopt-1

struct Small { size_t val; }
struct Large { size_t a, b, c, d, e, f, g, h; }

int byValue(ubyte ub, ushort us, uint ui, ulong ul,
            float f, double d, real r,
            cdouble c, int delegate() dg, int function() fun,
            int[] slice, float[int] aa, ubyte[16] fa,
            float4 f4, double4 d4,
            Small small, Large large,
            TypeInfo_Class ti, typeof(null) np)
{
// CDB: bp `args_cdb.d:27`
// CDB: g
    // arguments implicitly passed by reference aren't shown if unused
    float cim = c.im + fa[7] + dg() + small.val + large.a;
    return 1;
// CHECK: !args_cdb::byValue
// CDB: dv /t

// CHECK: unsigned char ub = 0x01
// CHECK: unsigned short us = 2
// CHECK: unsigned int ui = 3
// CHECK: unsigned int64 ul = 4
// CHECK: float f = 5
// CHECK: double d = 6
// CHECK: double r = 7
// x64: cdouble * c =
// x86: cdouble c =
// x64: int delegate() * dg =
// x86: int delegate() dg =
// CHECK: <function> * fun = {{0x[0-9a-f`]*}}
// x86: struct int[] slice =
// CHECK: struct float[int] aa =
// x64: unsigned char (*)[16] fa
// x86: unsigned char [16] fa
// x86: float [4] f4 = float [4]
// x86: double [4] d4 = double [4]
// CHECK: args_cdb::Small small
// x64: args_cdb::Large * large
// x86: args_cdb::Large large
// CHECK: struct object::TypeInfo_Class * ti = {{0x[0-9a-f`]*}}
// CHECK: void * np = {{0x[0`]*}}

// params emitted as locals (listed after params) for Win64:
// x64: struct int[] slice
// x64: float [4] f4 = float [4]
// x64: double [4] d4 = double [4]

// check arguments with indirections
// CDB: ?? c
// CHECK: cdouble
// CHECK-NEXT: re : 8
// CHECK-NEXT: im : 9

// CDB: ?? dg
// CHECK: int delegate()
// CHECK-NEXT: context :
// CHECK-NEXT: funcptr :
// CHECK-SAME: args_cdb.main.__lambda

// CDB: ?? slice
// CHECK: struct int[]
// CHECK-NEXT: length : 2
// CHECK-NEXT: ptr :
// CHECK-SAME: 0n10

// CDB: ?? (*fa)[1]
// x64: unsigned char 0x0e
// no-x86: would be fa[1], but displays garbage anyway

// CDB: ?? f4[1]
// CHECK: float 16

// CDB: ?? d4[1]
// CHECK: double 17

// CDB: ?? small
// CHECK: args_cdb::Small
// x64-NEXT: val : 0x12
// no-x86-NEXT: val : 0x12 (displays garbage)

// CDB: ?? large
// CHECK: args_cdb::Large
// x64-NEXT: a : 0x13
// no-x86-NEXT: a : 0x13 (displays garbage)

// CDB: ?? ti
// CHECK: object::TypeInfo_Class
// CHECK-NEXT: m_init : byte[]
}

int byPtr(ubyte* ub, ushort* us, uint* ui, ulong* ul,
          float* f, double* d, real* r,
          cdouble* c, int delegate()* dg, int function()* fun,
          int[]* slice, float[int]* aa, ubyte[16]* fa,
          float4* f4, double4* d4,
          Small* small, Large* large,
          TypeInfo_Class* ti, typeof(null)* np)
{
// CDB: bp `args_cdb.d:115`
// CDB: g
    return 3;
// CHECK: !args_cdb::byPtr
// CDB: dv /t
// CDB: ?? *ub
// CHECK: unsigned char 0x01
// CDB: ?? *us
// CHECK: unsigned short 2
// CDB: ?? *ui
// CHECK: unsigned int 3
// CDB: ?? *ul
// CHECK: unsigned int64 4
// CDB: ?? *f
// CHECK: float 5
// CDB: ?? *d
// CHECK: double 6
// CDB: ?? *r
// CHECK: double 7
// CDB: ?? *c
// CHECK: cdouble
// CHECK-NEXT: re : 8
// CHECK-NEXT: im : 9
// CDB: ?? *dg
// CHECK: int delegate()
// CHECK-NEXT: context :
// CHECK-NEXT: funcptr :
// CHECK-SAME: args_cdb.main.__lambda
// CDB: ?? *fun
// CHECK: <function> *
// CDB: ?? *slice
// CHECK: struct int[]
// CHECK-NEXT: length : 2
// CHECK-NEXT: ptr :
// CHECK-SAME: 0n10
// CDB: ?? *aa
// CHECK: struct float[int]
// CDB: ?? (*fa)[1]
// CHECK: unsigned char 0x0e
// CDB: ?? (*f4)[1]
// CHECK: float 16
// CDB: ?? (*d4)[2]
// CHECK: double 17
// CDB: ?? *small
// CHECK: struct args_cdb::Small
// CHECK-NEXT: val : 0x12
// CDB: ?? *large
// CHECK: struct args_cdb::Large
// CHECK-NEXT: a : 0x13
// CHECK-NEXT: b :
// CDB: ?? *ti
// CHECK: struct object::TypeInfo_Class
// CHECK-NEXT: m_init : byte[]
// CDB: ?? *np
// CHECK: void * {{0x[0`]*}}
}

int byRef(ref ubyte ub, ref ushort us, ref uint ui, ref ulong ul,
          ref float f, ref double d, ref real r,
          ref cdouble c, ref int delegate() dg, ref int function() fun,
          ref int[] slice, ref float[int] aa, ref ubyte[16] fa,
          ref float4 f4, ref double4 d4,
          ref Small small, ref Large large,
          ref TypeInfo_Class ti, ref typeof(null) np)
{
// CDB: bp `args_cdb.d:180`
// CDB: g
// CHECK: !args_cdb::byRef

// CDB: dv /t
// cdb displays references as pointers
// CDB: ?? *ub
// CHECK: unsigned char 0x01
// CDB: ?? *us
// CHECK: unsigned short 2
// CDB: ?? *ui
// CHECK: unsigned int 3
// CDB: ?? *ul
// CHECK: unsigned int64 4
// CDB: ?? *f
// CHECK: float 5
// CDB: ?? *d
// CHECK: double 6
// CDB: ?? *r
// CHECK: double 7
// CDB: ?? *c
// CHECK: cdouble
// CHECK-NEXT: re : 8
// CHECK-NEXT: im : 9
// CDB: ?? *dg
// CHECK: int delegate()
// CHECK-NEXT: context :
// CHECK-NEXT: funcptr : {{0x[0-9a-f`]*}}
// CHECK-SAME: args_cdb.main.__lambda
// CDB: ?? *fun
// CHECK: <function> * {{0x[0-9a-f`]*}}
// CDB: ?? *slice
// CHECK: struct int[]
// CHECK-NEXT: length : 2
// CHECK-NEXT: ptr : {{0x[0-9a-f`]*}} -> 0n10
// CDB: ?? (*fa)[1]
// CHECK: unsigned char 0x0e
// CDB: ?? *aa
// CHECK: struct float[int]
// CDB: ?? (*f4)[1]
// CHECK: float 16
// CDB: ?? (*d4)[2]
// CHECK: double 17
// CDB: ?? *small
// CHECK: struct args_cdb::Small
// CHECK-NEXT: val : 0x12
// CDB: ?? *large
// CHECK: struct args_cdb::Large
// CHECK-NEXT: a : 0x13
// CHECK-NEXT: b :
// CDB: ?? *ti
// CHECK: struct object::TypeInfo_Class * {{0x[0-9a-f`]*}}
// CHECK-NEXT: m_init : byte[]
// CDB: ?? *np
// no-CHECK: void * {{0x[0`]*}} (not available)

    // needs access to references to actually generate debug info
    float cim = c.im + fa[7] + dg() + fun() + slice.length + aa.length + f4[0] + d4[1] +
                small.val + large.a + ti.initializer.length;

    ub++;
    us++;
    ui++;
    ul++;
    f++;
    d++;
    r++;
    c = c + 1;
    dg = (){ return 1; };
    fun = (){ return 2; };
    slice[0]++;
    aa[12]++;
    fa[0]++;
    f4.array[0] = f4.array[0] + 1;
    d4.array[0] = d4.array[0] + 1;
    small.val++;
    large.a++;
    ti = typeid(TypeInfo);
    np = null;
    return 2;
}

int main()
{
    ubyte ub = 1;
    ushort us = 2;
    uint ui = 3;
    ulong ul = 4;
    float f = 5;
    double d = 6;
    real r = 7;
    cdouble c = 8 + 9i;
    int delegate() dg = (){ return 3;};
    int function() fun = (){ return 4; };
    int[] slice = [10, 11];
    float[int] aa; aa[12] = 13;
    ubyte[16] fa; fa[] = 14;
    float4 f4 = 16;
    double4 d4 = 17;
    Small small = Small(18);
    Large large = Large(19);
    TypeInfo_Class ti = typeid(TypeInfo);
    typeof(null) np = null;
    
    byValue(ub, us, ui, ul, f, d, r, c, dg, fun, slice, aa, fa, f4, d4, small, large, ti, np);
    byPtr(&ub, &us, &ui, &ul, &f, &d, &r, &c, &dg, &fun, &slice, &aa, &fa, &f4, &d4, &small, &large, &ti, &np);
    byRef(ub, us, ui, ul, f, d, r, c, dg, fun, slice, aa, fa, f4, d4, small, large, ti, np);

    return 0;
}
// CDB: q
// CHECK: quit
