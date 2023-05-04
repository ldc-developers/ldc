
// REQUIRES: gdb
// This test fails due to newer version of GDB, see https://github.com/ldc-developers/ldc/issues/4389
// XFAIL: FreeBSD

// RUN: %ldc %_gdb_dflags -g -of=%t %s
// RUN: sed -e "/^\\/\\/ GDB:/!d" -e "s,// GDB:,," %s >%t.gdb
// RUN: gdb %t --batch -x %t.gdb >%t.out 2>&1
// RUN: FileCheck %s -check-prefix=CHECK < %t.out
module classtypes_gdb;

class uv {
    uint i;
}

class xyz : uv {
    float f;
    double d;

    this(uint i, float f) { this.i = i; this.f = f; }
}

// There are debug info issues with TLS variables when LDC is built within older environments (incl. Travis).
__gshared uv gvar;
static this() { gvar = new xyz(12, 34.56); }

int main() {
    xyz[4] sarr;
    xyz* ptr;
    xyz lvar;

    lvar = new xyz(99, 88.77);
    lvar.d = 624.351;
    sarr[2] = new xyz(2, 2.0);
    sarr[2].d = 0.987;
    ptr = &lvar;
    // BP

// GDB: b classtypes_gdb.d:37
// GDB: r
    return 0;
// CHECK: D main

// GDB: p lvar
// CHECK: xyz{{ *}}*)

// GDB: p *lvar
// CHECK: i = 99}
// CHECK-SAME: f = 88.7
// CHECK-SAME: d = 624.35

// GDB: p *ptr
// CHECK: xyz{{ *}}*)

// GDB: p **ptr
// CHECK: i = 99}
// CHECK-SAME: f = 88.7
// CHECK-SAME: d = 624.35

// GDB: p sarr
// CHECK: {0x0,{{ *}}0x0,{{ *}}0x{{[0-9a-f][0-9a-f]+}},{{ *}}0x0}

// GDB: p *sarr[2]
// CHECK: i = 2}
// CHECK-SAME: f = 2
// CHECK-SAME: d = 0.98

// GDB: p _D14classtypes_gdb4gvarCQw2uv
// CHECK: uv{{ *}}*)
// GDB: p *_D14classtypes_gdb4gvarCQw2uv
// CHECK: i = 12}{{$}}
}

// GDB: c
// GDB: q
// CHECK: exited normally
