// REQUIRES: Windows
// REQUIRES: cdb

// -g:
// RUN: %ldc -g -of=%t_g.exe %s
// RUN: sed -e "/^\\/\\/ CDB:/!d" -e "s,// CDB:,," %s \
// RUN:    | %cdb -snul -lines -y . %t_g.exe >%t_g.out
// RUN: FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-G -check-prefix=CHECK-G-%arch < %t_g.out

// -gc:
// RUN: %ldc -gc -of=%t_gc.exe %s
// RUN: sed -e "/^\\/\\/ CDB:/!d" -e "s,// CDB:,," %s \
// RUN:    | %cdb -snul -lines -y . %t_gc.exe >%t_gc.out
// RUN: FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-GC -check-prefix=CHECK-GC-%arch < %t_gc.out

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
// CDB: bp0 /1 `args_cdb.d:34`
// CDB: g
// CHECK: Breakpoint 0 hit
// CHECK-G:  !args_cdb.byValue
// CHECK-GC: !args_cdb::byValue

// CDB: dv /t

// CHECK: unsigned char ub = 0x01
// CHECK: unsigned short us = 2
// CHECK: unsigned int ui = 3
// CHECK: unsigned int64 ul = 4
// CHECK: float f = 5
// CHECK: double d = 6
// CHECK: double r = 7

/*
 * On Win64, by-value params > 64-bits are declared as DI locals, not parameters.
 * Locals are listed after params in cdb's `dv /t`, so don't check the order (-DAG).
 */
// CHECK-DAG: cdouble c =
// CHECK-DAG: int delegate() dg =
// CHECK-DAG: <function> * fun = {{0x[0-9a-f`]*}}
// CHECK-G-DAG:  struct int[] slice =
// CHECK-GC-DAG: struct slice<int> slice =
// CHECK-G-DAG:  struct float[int] aa =
// CHECK-GC-DAG: struct associative_array<int, float> aa =
// CHECK-DAG: unsigned char [16] fa =
// CHECK-DAG: float [4] f4 =
// CHECK-DAG: double [4] d4 =
// CHECK-G-DAG:  struct args_cdb.Small small =
// CHECK-GC-DAG: struct args_cdb::Small small =
/*
 * On Win64, `large` is emitted as reference, according to CodeViewDebug::calculateRanges()
 * in llvm/lib/CodeGen/AsmPrinter/CodeViewDebug.cpp because of a CodeView limitation, as
 * the pointer to the hidden copy is on the stack and not in a register.
 * See the byValueShort() test below.
 */
// CHECK-G-x86-DAG:  struct args_cdb.Large large =
// CHECK-GC-x86-DAG: struct args_cdb::Large large =
// CHECK-G-x64-DAG:  struct args_cdb.Large * large =
// CHECK-GC-x64-DAG: struct args_cdb::Large * large =
// CHECK-G-DAG:  struct object.TypeInfo_Class * ti = {{0x[0-9a-f`]*}}
// CHECK-GC-DAG: struct object::TypeInfo_Class * ti = {{0x[0-9a-f`]*}}
// CHECK-DAG: void * np = {{0x[0`]*}}

// check arguments with indirections
// CDB: ?? c
// CHECK: > struct cdouble
// CHECK-NEXT: re : 8
// CHECK-NEXT: im : 9

// CDB: ?? dg
// CHECK: int delegate()
// CHECK-NEXT: ptr :
// CHECK-NEXT: funcptr :
// CHECK-G-SAME: args_cdb.main.__lambda
// CHECK-GC-SAME: args_cdb::main::__lambda

// CDB: ?? slice
// CHECK-G:  struct int[]
// CHECK-GC: struct slice<int>
// CHECK-NEXT: length : 2
// CHECK-NEXT: ptr :
// CHECK-SAME: 0n10

// CDB: ?? fa[1]
// CHECK: unsigned char 0x0e

// CDB: ?? f4[1]
// CHECK: float 16

// CDB: ?? d4[1]
// CHECK: double 17

// CDB: ?? small
// CHECK-G:  args_cdb.Small
// CHECK-GC: args_cdb::Small
// CHECK-NEXT: val : 0x12

// CDB: ?? large
// CHECK-G:  args_cdb.Large
// CHECK-GC: args_cdb::Large
// CHECK-NEXT: a : 0x13

// CDB: ?? ti
// CHECK-G:  object.TypeInfo_Class
// CHECK-GC: object::TypeInfo_Class
// CHECK-G-NEXT:  m_init : byte[]
// CHECK-GC-NEXT: m_init : slice<byte>

    // arguments implicitly passed by reference aren't shown if unused
    float cim = c.im + fa[7] + dg() + small.val + large.a;
    return 1;
}

/*
 * On Win64, it makes a difference whether an argument > 64 bits (rewritten by
 * ImplicitByvalRewrite) is one the first 4 LL args (pointer to hidden copy in
 * register or spilled to stack).
 * The latter case is tested above, this tests the register case.
 */
size_t byValueShort(Large large)
{
// CDB: bp1 /1 `args_cdb.d:138`
// CDB: g
// CHECK: Breakpoint 1 hit
// CHECK-G:  !args_cdb.byValueShort
// CHECK-GC: !args_cdb::byValueShort

// CDB: dv /t
// CHECK-G:  struct args_cdb.Large large = struct args_cdb.Large
// CHECK-GC: struct args_cdb::Large large = struct args_cdb::Large

// CDB: ?? large
// CHECK-G:  args_cdb.Large
// CHECK-GC: args_cdb::Large
// CHECK-NEXT: a : 0x13

    return large.a;
}

int byPtr(ubyte* ub, ushort* us, uint* ui, ulong* ul,
          float* f, double* d, real* r,
          cdouble* c, int delegate()* dg, int function()* fun,
          int[]* slice, float[int]* aa, ubyte[16]* fa,
          float4* f4, double4* d4,
          Small* small, Large* large,
          TypeInfo_Class* ti, typeof(null)* np)
{
// CDB: bp2 /1 `args_cdb.d:164`
// CDB: g
// CHECK: Breakpoint 2 hit
// CHECK-G:  !args_cdb.byPtr
// CHECK-GC: !args_cdb::byPtr

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
// CHECK-NEXT: ptr :
// CHECK-NEXT: funcptr :
// CHECK-G-SAME: args_cdb.main.__lambda
// CHECK-GC-SAME: args_cdb::main::__lambda
// CDB: ?? *fun
// CHECK: <function> *
// CDB: ?? *slice
// CHECK-G:  struct int[]
// CHECK-GC: struct slice<int>
// CHECK-NEXT: length : 2
// CHECK-NEXT: ptr :
// CHECK-SAME: 0n10
// CDB: ?? *aa
// CHECK-G:  struct float[int]
// CHECK-GC: struct associative_array<int, float>
// CDB: ?? (*fa)[1]
// CHECK: unsigned char 0x0e
// CDB: ?? (*f4)[1]
// CHECK: float 16
// CDB: ?? (*d4)[2]
// CHECK: double 17
// CDB: ?? *small
// CHECK-G:  struct args_cdb.Small
// CHECK-GC: struct args_cdb::Small
// CHECK-NEXT: val : 0x12
// CDB: ?? *large
// CHECK-G:  struct args_cdb.Large
// CHECK-GC: struct args_cdb::Large
// CHECK-NEXT: a : 0x13
// CHECK-NEXT: b :
// CDB: ?? *ti
// CHECK-G:  struct object.TypeInfo_Class
// CHECK-GC: struct object::TypeInfo_Class
// CHECK-G-NEXT:  m_init : byte[]
// CHECK-GC-NEXT: m_init : slice<byte>
// CDB: ?? *np
// CHECK: void * {{0x[0`]*}}

    return 3;
}

int byRef(ref ubyte ub, ref ushort us, ref uint ui, ref ulong ul,
          ref float f, ref double d, ref real r,
          ref cdouble c, ref int delegate() dg, ref int function() fun,
          ref int[] slice, ref float[int] aa, ref ubyte[16] fa,
          ref float4 f4, ref double4 d4,
          ref Small small, ref Large large,
          ref TypeInfo_Class ti, ref typeof(null) np)
{
// CDB: bp3 /1 `args_cdb.d:240`
// CDB: g
// CHECK: Breakpoint 3 hit
// CHECK-G:  !args_cdb.byRef
// CHECK-GC: !args_cdb::byRef

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
// CHECK-NEXT: ptr :
// CHECK-NEXT: funcptr : {{0x[0-9a-f`]*}}
// CHECK-G-SAME: args_cdb.main.__lambda
// CHECK-GC-SAME: args_cdb::main::__lambda
// CDB: ?? *fun
// CHECK: <function> * {{0x[0-9a-f`]*}}
// CDB: ?? *slice
// CHECK-G:  struct int[]
// CHECK-GC: struct slice<int>
// CHECK-NEXT: length : 2
// CHECK-NEXT: ptr : {{0x[0-9a-f`]*}} -> 0n10
// CDB: ?? (*fa)[1]
// CHECK: unsigned char 0x0e
// CDB: ?? *aa
// CHECK-G:  struct float[int]
// CHECK-GC: struct associative_array<int, float>
// CDB: ?? (*f4)[1]
// CHECK: float 16
// CDB: ?? (*d4)[2]
// CHECK: double 17
// CDB: ?? *small
// CHECK-G:  struct args_cdb.Small
// CHECK-GC: struct args_cdb::Small
// CHECK-NEXT: val : 0x12
// CDB: ?? *large
// CHECK-G:  struct args_cdb.Large
// CHECK-GC: struct args_cdb::Large
// CHECK-NEXT: a : 0x13
// CHECK-NEXT: b :
// CDB: ?? *ti
// CHECK-G:  struct object.TypeInfo_Class * {{0x[0-9a-f`]*}}
// CHECK-GC: struct object::TypeInfo_Class * {{0x[0-9a-f`]*}}
// CHECK-G-NEXT:  m_init : byte[]
// CHECK-GC-NEXT: m_init : slice<byte>
// CDB: ?? *np
/*
 * On Win32 (and Win64 since LLVM 9), `np` is somehow not available.
 * C HECK: void * {{0x[0`]*}}
 */

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
    byValueShort(large);
    byPtr(&ub, &us, &ui, &ul, &f, &d, &r, &c, &dg, &fun, &slice, &aa, &fa, &f4, &d4, &small, &large, &ti, &np);
    byRef(ub, us, ui, ul, f, d, r, c, dg, fun, slice, aa, fa, f4, d4, small, large, ti, np);

    return 0;
}
// CDB: q
// CHECK: quit
