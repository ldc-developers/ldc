
// REQUIRES: Windows
// REQUIRES: cdb
// RUN: %ldc -g -of=%t.exe %s
// RUN: sed -e "/^\\/\\/ CDB:/!d" -e "s,// CDB:,," %s \
// RUN:    | %cdb -snul -lines -y . %t.exe >%t.out
// RUN: FileCheck %s < %t.out

// modulename explicitly unspecified to check implicit function name when breaking
void main()
{
    basic_types();
}

int basic_types()
{
    char c = 'a';
    wchar wc = 'b';
    dchar dc = 'c';
    byte b = 1;
    ubyte ub = 2;
    short s = 3;
    ushort us = 4;
    int i = 5;
    uint ui = 6;
    long l = 7;
    ulong ul = 8;

    float f = 9;
    double d = 10;
    real r = 11;

    ifloat iflt = 12i;
    idouble id = 13i;
    ireal ir = 14i;
    cfloat cf = 15 + 16i;
    cdouble cd = 17 + 18i;
    creal cr = 19 + 20i;
    typeof(null) np = null;

    c = c;
// CDB: ld basictypes_cdb*
// CDB: bp0 /1 `basictypes_cdb.d:43`
// CDB: g
// CHECK: Breakpoint 0 hit
// CHECK: !basictypes_cdb.basic_types

// enable case sensitive symbol lookup
// CDB: .symopt-1
// CDB: dv /t
// CHECK: char c = 0n97 'a'
// wc: UTF16 not supported by cvd, works in VS
// dc: UTF32 not supported by cvd, works in VS
// CHECK: char b = 0n1 ''
// CHECK: unsigned char ub = 0x02 ''
// CHECK: short s = 0n3
// CHECK: unsigned short us = 4
// CHECK: int i = 0n5
// CHECK: unsigned int ui = 6
// CHECK: int64 l = 0n7
// CHECK: unsigned int64 ul = 8
// CHECK: float f = 9
// CHECK: double d = 10
// CHECK: double r = 11
// CHECK: float iflt = 12
// CHECK: double id = 13
// CHECK: double ir = 14
// CHECK: struct cfloat cf
// CHECK: struct cdouble cd
// CHECK: struct creal cr
// CHECK: void * np = 0x{{[0`]+}}
// CDB: ?? cf
// CHECK: +0x000 re : 15
// CHECK: +0x004 im : 16
// CDB: ?? cd
// CHECK: +0x000 re : 17
// CHECK: +0x008 im : 18
// CDB: ?? cr
// CHECK: +0x000 re : 19
// CHECK: +0x008 im : 20

    return 1;
}
// CDB: q
// CHECK: quit
