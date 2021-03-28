// REQUIRES: Windows
// REQUIRES: cdb

// RUN: %ldc -g -of=%t.exe %s
// RUN: sed -e "/^\\/\\/ CDB:/!d" -e "s,// CDB:,," %s \
// RUN:    | %cdb -snul -lines -y . %t.exe >%t.out
// RUN: FileCheck %s < %t.out

enum E : byte { x, y, z }

struct S { E e; }

void foo(E e, S s)
{
// CDB: ld /f enums_cdb*
// enable case sensitive symbol lookup
// CDB: .symopt-1
// CDB: bp0 /1 `enums_cdb.d:18`
// CDB: g
// CHECK: Breakpoint 0 hit
// CHECK: !enums_cdb.foo

// CDB: ?? e
// CHECK: enums_cdb.E y

// CDB: ?? s.e
// CHECK-NEXT: enums_cdb.E z
}

void main()
{
    foo(E.y, S(E.z));
}

// CDB: q
// CHECK: quit:
