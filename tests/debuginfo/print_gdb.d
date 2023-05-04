// REQUIRES: atleast_gdb80

// This test fails due to newer version of GDB, see https://github.com/ldc-developers/ldc/issues/4389
// XFAIL: FreeBSD

// RUN: %ldc %_gdb_dflags -I%S -g -of=%t %s %S/inputs/import_a.d %S/inputs/import_b.d
// RUN: sed -e "/^\\/\\/ GDB:/!d" -e "s,// GDB:,," %s >%t.gdb
// RUN: env LANG=C gdb %t --batch -x %t.gdb >%t.out 2>&1
// RUN: FileCheck %s -check-prefix=CHECK < %t.out
module print_gdb;

import inputs.import_a;

__gshared int globVal = 987;

enum eA { ABC = 2, ZYX }

struct sA {
    static int someVal = 246;
}

struct sB {
    uint k = 9;
    uint memberFunc(uint a) { return k*k+a; }

    static staticFunc(uint b) { return b * 2; }
}

class cC {
    char c = '0';
    char classMemberFunc(byte a) { return cast(char)(cast(byte)c+a); }

    static classStaticFunc(byte b) { return cast(char)(cast(byte)'a' + b); }

    mixin mix;
}

struct templatedStruct(T) {
    T z;
    T pal(T m) { return z * m; }
}

mixin template mix()
{
    uint mixedVal = 5;
}

double foo(double plus)
{
    return 42.13 + plus;
}

void main()
{
// GDB: b _Dmain
// GDB: r

    uint n = 5;
    n = globVal; // reference every global symbol in main() to have them emitted
    n = sA.someVal;
    n = cast(int) a_sA.statChar;

    foo(242.0);
    bar();

    // BP
// GDB: b print_gdb.d:66
// GDB: c
// GDB: p globVal
// CHECK: = 987
// GDB: p sA.someVal
// CHECK: = 246
// GDB: p a_sA.statChar
// CHECK: = 67 'C'
// GDB: p foo(0.12)
// CHECK: = 42.25

    sB strB;
    strB.k = 12;
    strB.k = strB.memberFunc(2);
    strB.k = strB.staticFunc(3); // k = 6

    eA e = eA.ZYX;

    import inputs.import_b;
    b_Glob = 99.88;
    inputs.import_b.b_cA.staticVal = 78;

    // BP
// GDB: b print_gdb.d:89
// GDB: c
// GDB: p strB
// GDB: p strB.memberFunc(4)
// CHECK: = 40
// GDB: p sB.staticFunc(44)
// CHECK: = 88
// GDB: whatis eA
// CHECK: type = print_gdb.eA
// GDB: whatis print_gdb.eA
// CHECK: type = print_gdb.eA
// GDB: p b_Glob
// CHECK: = 99.8

    cC clsC = new cC;
    clsC.classMemberFunc(2);
    cC.classStaticFunc(4);
    clsC.mixedVal++;

        // BP
// GDB: b print_gdb.d:109
// GDB: c
// GDB: p *clsC
// GDB: p clsC.classMemberFunc(6)
// CHECK: = 54 '6'
// GDB: p clsC.classStaticFunc(4)
// CHECK: = 101 'e'
// GDB: p clsC.mixedVal
// CHECK: = 6

//     class cD { static int staticVal = 741; }
    // FIXME: GDB doesn't support access to nested type declarations yet

    templatedStruct!int tsI;
    templatedStruct!float tsF;

    // BP
// GDB: b print_gdb.d:126
// GDB: c
// GDB: whatis tsF
// CHECK: type = print_gdb.templatedStruct
}

// GDB: c
// GDB: q
// CHECK: exited normally
