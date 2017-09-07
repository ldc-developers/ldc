// REQUIRES: Windows
// REQUIRES: cdb
// RUN: %ldc -g -of=%t.exe %s
// RUN: sed -e "/^\\/\\/ CDB:/!d" -e "s,// CDB:,," %s \
// RUN:    | %cdb -snul -lines -y . %t.exe >%t.out
// RUN: FileCheck %s -check-prefix=CHECK -check-prefix=%arch < %t.out

// CDB: ld /f nested_cdb*
// enable case sensitive symbol lookup
// CDB: .symopt-1

void encloser(int arg0, ref int arg1)
{
    int enc_n = 123;
// CDB: bp `nested_cdb.d:15`
// CDB: g
// CDB: dv /t
// CHECK: int arg0 = 0n1
// arg1 is missing
// CHECK-NEXT: int enc_n = 0n123
    enc_n += arg1;

    void nested(int nes_i)
    {
        int blub = arg0 + arg1 + enc_n;
// CDB: bp `nested_cdb.d:26`
// CDB: g
// CDB: dv /t
// CHECK: int arg0 = 0n1
// arg1 is missing
// CHECK-NEXT: int enc_n = 0n125
        arg0 = arg1 = enc_n = nes_i;
// CDB: bp `nested_cdb.d:33`
// CDB: g
// CDB: dv /t
// CHECK: int arg0 = 0n456
// arg1 is missing
// CHECK-NEXT: int enc_n = 0n456
    }

    nested(456);
// CDB: bp `nested_cdb.d:42`
// CDB: g
// CDB: dv /t
// CHECK: int arg0 = 0n456
// arg1 is missing
// CHECK-NEXT: int enc_n = 0n456
}

void main()
{
    int arg1 = 2;
    encloser(1, arg1);
}

// CDB: q
// CHECK: quit
