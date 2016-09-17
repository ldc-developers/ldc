// REQUIRES: atleast_llvm308
// REQUIRES: Windows
// REQUIRES: cdb
// RUN: %ldc -g -of=%t.exe %s
// RUN: sed -e "/^\\/\\/ CDB:/!d" -e "s,// CDB:,," %s \
// RUN:    | %cdb -snul -lines -y . %t.exe >%t.out
// RUN: FileCheck %s -check-prefix=CHECK -check-prefix=%arch < %t.out

class BaseClass
{
    uint baseMember = 3;
}

class DerivedClass : BaseClass
{
    uint derivedMember = 7;
}

int main(string[] args)
{
    auto dc = new DerivedClass;
// CDB: ld /f cvbaseclass*
// enable case sensitive symbol lookup
// CDB: .symopt-1
// CDB: bp `cvbaseclass.d:27`
// CDB: g
    return 0;
// CHECK: !D main

// CDB: ?? dc
// cdb doesn't show base class info, but lists their members
// CHECK: DerivedClass
// CHECK: baseMember{{ *: *3}}
// verify baseMember is not listed twice
// CHECK-NEXT: derivedMember{{ *: *7}}
}

// CDB: q
// CHECK: quit:
