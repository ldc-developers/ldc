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
// CDB: bp `scopes_cdb.d:27`
// CDB: g
// CHECK-G:  !scopes_cdb.Template!1.foo+
// CHECK-GC: !scopes_cdb::Template<1>::foo+
    }
}

extern (C++, cppns)
{
    void cppFoo()
    {
// CDB: bp `scopes_cdb.d:38`
// CDB: g
// CHECK-G:  !scopes_cdb.cppns.cppFoo+
// CHECK-GC: !scopes_cdb::cppns::cppFoo+
    }
}

void templatedFoo(int N)()
{
// CDB: bp `scopes_cdb.d:47`
// CDB: g
// CHECK-G:  !scopes_cdb.templatedFoo!2+
// CHECK-GC: !scopes_cdb::templatedFoo<2>+
}

mixin template Mixin(T)
{
    T mixedInField;
    void mixedInFoo()
    {
// CDB: bp `scopes_cdb.d:58`
// CDB: g
// CHECK-G:  !scopes_cdb.S.mixedInFoo+
// CHECK-GC: !scopes_cdb::S::mixedInFoo+
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
// CDB: bp `scopes_cdb.d:86`
// CDB: g
// CHECK-G:  !scopes_cdb.test.TemplatedNestedStruct!(S, 3).foo+
// CHECK-GC: !scopes_cdb::test::TemplatedNestedStruct<S, 3>::foo+
        }
    }

    TemplatedNestedStruct!(S, 3) tns;
    tns.foo();

    static struct NestedStruct
    {
        int field;
        void foo()
        {
// CDB: bp `scopes_cdb.d:101`
// CDB: g
// CHECK-G:  !scopes_cdb.test.NestedStruct.foo+
// CHECK-GC: !scopes_cdb::test::NestedStruct::foo+
        }
    }

    NestedStruct ns;
    ns.foo();

// CDB: bp `scopes_cdb.d:111`
// CDB: g
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
