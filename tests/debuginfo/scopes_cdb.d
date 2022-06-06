// REQUIRES: Windows
// REQUIRES: cdb

// -g:
// RUN: %ldc -g -of=%t_g.exe %s
// RUN: sed -e "/^\\/\\/ CDB:/!d" -e "s,// CDB:,," %s \
// RUN:    | %cdb -snul -lines -y . %t_g.exe >%t_g.out
// RUN: FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-G < %t_g.out

// -gc:
// RUN: %ldc -gc -of=%t_gc.exe %s
// RUN: sed -e "/^\\/\\/ CDB:/!d" -e "s,// CDB:,," %s \
// RUN:    | %cdb -snul -lines -y . %t_gc.exe >%t_gc.out
// RUN: FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-GC < %t_gc.out

module scopes_cdb;

// CDB: ld /f scopes_cdb*
// enable case sensitive symbol lookup
// CDB: .symopt-1

template Template(int N)
{
    int[N] field;
    void foo()
    {
// CDB: bp0 /1 `scopes_cdb.d:27`
// CDB: g
// CHECK: Breakpoint 0 hit
// CHECK-G:  !scopes_cdb.Template!1.foo
// CHECK-GC: !scopes_cdb::Template<1>::foo
    }
}

extern (C++, cppns)
{
    void cppFoo()
    {
// CDB: bp1 /1 `scopes_cdb.d:39`
// CDB: g
// CHECK: Breakpoint 1 hit
// CHECK-G:  !scopes_cdb.cppns.cppFoo
// CHECK-GC: !scopes_cdb::cppns::cppFoo
    }
}

void templatedFoo(int N)()
{
// CDB: bp2 /1 `scopes_cdb.d:49`
// CDB: g
// CHECK: Breakpoint 2 hit
// CHECK-G:  !scopes_cdb.templatedFoo!2
// CHECK-GC: !scopes_cdb::templatedFoo<2>
}

mixin template Mixin(T)
{
    // test https://github.com/ldc-developers/ldc/issues/2937 while at it
    static foreach (i; 0 .. 1)
    {
        static struct MixedInStruct { T field; }
        MixedInStruct mixedInField;
        void mixedInFoo()
        {
            MixedInStruct local;
// CDB: bp3 /1 `scopes_cdb.d:66`
// CDB: g
// CHECK: Breakpoint 3 hit
// CHECK-G:  !scopes_cdb.S.mixedInFoo
// CHECK-GC: !scopes_cdb::S::mixedInFoo
// CDB: dv /t
// CHECK-G:  struct scopes_cdb.S.MixedInStruct local =
// CHECK-GC: struct scopes_cdb::S::MixedInStruct local =
        }
    }
}

struct S
{
    mixin Mixin!double;
}

void test()
{
    Template!1.foo();

    cppFoo();

    templatedFoo!2();

    S s;
    s.mixedInFoo();

    static struct TemplatedNestedStruct(T, int N)
    {
        T[N] field;
        void foo()
        {
// CDB: bp4 /1 `scopes_cdb.d:99`
// CDB: g
// CHECK: Breakpoint 4 hit
// CHECK-G:  !scopes_cdb.test.TemplatedNestedStruct!(S, 3).foo
// CHECK-GC: !scopes_cdb::test::TemplatedNestedStruct<S, 3>::foo
        }
    }

    TemplatedNestedStruct!(S, 3) tns;
    tns.foo();

    static struct NestedStruct
    {
        int field;
        void foo()
        {
// CDB: bp5 /1 `scopes_cdb.d:115`
// CDB: g
// CHECK: Breakpoint 5 hit
// CHECK-G:  !scopes_cdb.test.NestedStruct.foo
// CHECK-GC: !scopes_cdb::test::NestedStruct::foo
        }
    }

    NestedStruct ns;
    ns.foo();

// CDB: bp6 /1 `scopes_cdb.d:126`
// CDB: g
// CHECK: Breakpoint 6 hit
// CDB: dv /t
// CHECK-G:       struct scopes_cdb.S s =
// CHECK-GC:      struct scopes_cdb::S s =
// CHECK-G-NEXT:  struct scopes_cdb.test.TemplatedNestedStruct!(S, 3) tns =
// CHECK-GC-NEXT: struct scopes_cdb::test::TemplatedNestedStruct<S, 3> tns =
// CHECK-G-NEXT:  struct scopes_cdb.test.NestedStruct ns =
// CHECK-GC-NEXT: struct scopes_cdb::test::NestedStruct ns =
}

void main()
{
    test();
}

// CDB: q
// CHECK: quit
