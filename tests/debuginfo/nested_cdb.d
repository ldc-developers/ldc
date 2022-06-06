// REQUIRES: Windows
// REQUIRES: cdb
// RUN: %ldc -g -of=%t.exe %s
// RUN: sed -e "/^\\/\\/ CDB:/!d" -e "s,// CDB:,," %s \
// RUN:    | %cdb -snul -lines -y . %t.exe >%t.out
// RUN: FileCheck %s < %t.out

// CDB: ld /f nested_cdb*
// enable case sensitive symbol lookup
// CDB: .symopt-1

void encloser(int arg0, ref int arg1)
{
    int enc_n = 123;
// CDB: bp0 /1 `nested_cdb.d:15`
// CDB: g
// CHECK: Breakpoint 0 hit

// CDB: dv /t
// CHECK: int arg0 = 0n1
// (cdb displays references as pointers)
// CHECK-NEXT: int * arg1 = {{0x[0-9a-f`]*}}
// CHECK-NEXT: int enc_n = 0n123
// CDB: ?? *arg1
// CHECK: int 0n2
    enc_n += arg1;

    void nested(int nes_i)
    {
        int blub = arg0 + arg1 + enc_n;
// CDB: bp1 /1 `nested_cdb.d:31`
// CDB: g
// CHECK: Breakpoint 1 hit
// CDB: dv /t
// CHECK: int arg0 = 0n1
// CHECK-NEXT: int * arg1 = {{0x[0-9a-f`]*}}
// CHECK-NEXT: int enc_n = 0n125
// CDB: ?? *arg1
// CHECK: int 0n2
        arg0 = arg1 = enc_n = nes_i;
// CDB: bp2 /1 `nested_cdb.d:41`
// CDB: g
// CHECK: Breakpoint 2 hit
// CDB: dv /t
// CHECK: int arg0 = 0n456
// CHECK-NEXT: int * arg1 = {{0x[0-9a-f`]*}}
// CHECK-NEXT: int enc_n = 0n456
// CDB: ?? *arg1
// CHECK: int 0n456
    }

    nested(456);
// CDB: bp3 /1 `nested_cdb.d:53`
// CDB: g
// CHECK: Breakpoint 3 hit
// CDB: dv /t
// CHECK: int arg0 = 0n456
// CHECK-NEXT: int * arg1 = {{0x[0-9a-f`]*}}
// CHECK-NEXT: int enc_n = 0n456
// CDB: ?? *arg1
// CHECK: int 0n456
}

void main()
{
    int arg1 = 2;
    encloser(1, arg1);
}

// CDB: q
// CHECK: quit
