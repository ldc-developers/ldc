// REQUIRES: gdb
// RUN: %ldc -g -of=%t %s
// RUN: sed -e "/^\\/\\/ GDB:/!d" -e "s,// GDB:,," %s >%t.gdb
// RUN: gdb %t --batch -x %t.gdb >%t.out 2>&1
// RUN: FileCheck %s -check-prefix=CHECK < %t.out

// GDB: b _Dmain
// GDB: r

void encloser(int arg0, ref int arg1)
{
    int enc_n = 123;
// GDB: b 13
// GDB: c
// GDB: p arg0
// CHECK: $1 = 1
// GDB: p arg1
// CHECK: $2 = (int &) @{{0x[0-9a-f]*}}: 2
// GDB: p enc_n
// CHECK: $3 = 123
    enc_n += arg1;

    void nested(int nes_i)
    {
        int blub = arg0 + arg1 + enc_n;
// GDB: b 26
// GDB: c
// GDB: p arg0
// CHECK: $4 = 1
// GDB: p arg1
// CHECK: $5 = (int &) @{{0x[0-9a-f]*}}: 2
// GDB: p enc_n
// CHECK: $6 = 125
        arg0 = arg1 = enc_n = nes_i;
// GDB: b 35
// GDB: c
// GDB: p arg0
// CHECK: $7 = 456
// GDB: p arg1
// CHECK: $8 = (int &) @{{0x[0-9a-f]*}}: 456
// GDB: p enc_n
// CHECK: $9 = 456
    }

    nested(456);
// GDB: b 46
// GDB: c
// GDB: p arg0
// CHECK: $10 = 456
// GDB: p arg1
// CHECK: $11 = (int &) @{{0x[0-9a-f]*}}: 456
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
