// REQUIRES: Windows
// REQUIRES: cdb
// RUN: %ldc -g -of=%t.exe %s
// RUN: sed -e "/^\\/\\/ CDB:/!d" -e "s,// CDB:,," %s \
// RUN:    | %cdb -snul -lines -y . %t.exe >%t.out
// RUN: FileCheck %s -check-prefix=CHECK -check-prefix=%arch < %t.out

module scopes_cdb;

// CDB: ld /f scopes_cdb*
// enable case sensitive symbol lookup
// CDB: .symopt-1

template Template(int N)
{
    int[N] field;
    void foo()
    {
// CDB: bp `scopes_cdb.d:19`
// CDB: g
// CHECK: !scopes_cdb::Template!1::foo+
    }
}

extern (C++, cppns)
{
    void cppFoo()
    {
// CDB: bp `scopes_cdb.d:29`
// CDB: g
// CHECK: !scopes_cdb::cppns::cppFoo+
    }
}

void templatedFoo(int N)()
{
// CDB: bp `scopes_cdb.d:37`
// CDB: g
// CHECK: !scopes_cdb::templatedFoo!2+
}

mixin template Mixin(T)
{
    T mixedInField;
    void mixedInFoo()
    {
// CDB: bp `scopes_cdb.d:47`
// CDB: g
// CHECK: !scopes_cdb::S::mixedInFoo+
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
// CDB: bp `scopes_cdb.d:74`
// CDB: g
// CHECK: !scopes_cdb::test::TemplatedNestedStruct!(S, 3)::foo+
        }
    }

    TemplatedNestedStruct!(S, 3) tns;
    tns.foo();

    static struct NestedStruct
    {
        int field;
        void foo()
        {
// CDB: bp `scopes_cdb.d:88`
// CDB: g
// CHECK: !scopes_cdb::test::NestedStruct::foo+
        }
    }

    NestedStruct ns;
    ns.foo();

// CDB: bp `scopes_cdb.d:97`
// CDB: g
// CDB: dv /t
// CHECK:      struct scopes_cdb::S s =
// CHECK-NEXT: struct scopes_cdb::test::TemplatedNestedStruct!(S, 3) tns =
// CHECK-NEXT: struct scopes_cdb::test::NestedStruct ns =
}

void main()
{
    test();
}

// CDB: q
// CHECK: quit
