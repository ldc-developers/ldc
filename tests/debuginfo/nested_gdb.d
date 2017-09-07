// REQUIRES: gdb
// RUN: %ldc %_gdb_dflags -g -of=%t %s
// RUN: sed -e "/^\\/\\/ GDB:/!d" -e "s,// GDB:,," %s >%t.gdb
// RUN: gdb %t --batch -x %t.gdb >%t.out 2>&1
// RUN: FileCheck %s -check-prefix=CHECK < %t.out

void encloser(int arg0, ref int arg1)
{
    int enc_n = 123;
// GDB: b 10
// GDB: r
// GDB: p arg0
// CHECK: $1 = 1
// GDB: p arg1
// no-CHECK: $2 = 2 (<optimized out>)
// GDB: p enc_n
// CHECK: $3 = 123
    enc_n += arg1;

    void nested(int nes_i)
    {
        int blub = arg0 + arg1 + enc_n;
// GDB: b 23
// GDB: c
// GDB: p arg0
// CHECK: $4 = 1
// GDB: p arg1
// no-CHECK: $5 = 2 (<optimized out>)
// GDB: p enc_n
// CHECK: $6 = 125
        arg0 = arg1 = enc_n = nes_i;
// GDB: b 32
// GDB: c
// GDB: p arg0
// CHECK: $7 = 456
// GDB: p arg1
// no-CHECK: $8 = 456 (<optimized out>)
// GDB: p enc_n
// CHECK: $9 = 456
    }

    nested(456);
// GDB: b 43
// GDB: c
// GDB: p arg0
// CHECK: $10 = 456
// GDB: p arg1
// no-CHECK: $11 = 456 (<optimized out>)
// GDB: p enc_n
// CHECK: $12 = 456
}

void main()
{
    int arg1 = 2;
    encloser(1, arg1);
}

// GDB: c
// GDB: q
// CHECK: exited normally
