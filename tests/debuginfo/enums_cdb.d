// REQUIRES: Windows
// REQUIRES: cdb

// RUN: %ldc -g -of=%t.exe %s
// RUN: sed -e "/^\\/\\/ CDB:/!d" -e "s,// CDB:,," %s \
// RUN:    | %cdb -snul -lines -y . %t.exe >%t.out
// RUN: FileCheck %s < %t.out

enum E : byte { x, y, z }

enum D : double { d0, d1, d123 = 123.0 };

struct S { E e; D d; }

void foo(E e, D d, S s)
{
    auto bla = s; // somehow needed for s to show up...

// CDB: ld /f enums_cdb*
// enable case sensitive symbol lookup
// CDB: .symopt-1
// CDB: bp0 /1 `enums_cdb.d:22`
// CDB: g
// CHECK: Breakpoint 0 hit
// CHECK: !enums_cdb.foo

// CDB: ?? e
// CHECK: enums_cdb.E y

// CDB: ?? d
// CHECK-NEXT: double 1

// CDB: ?? s.e
// CHECK-NEXT: enums_cdb.E z

// CDB: ?? s.d
// CHECK-NEXT: double 123
}

void main()
{
    foo(E.y, D.d1, S(E.z, D.d123));
}

// CDB: q
// CHECK: quit:
