
// REQUIRES: Windows
// REQUIRES: cdb
// RUN: %ldc -g -of=%t.exe %s
// RUN: sed -e "/^\\/\\/ CDB:/!d" -e "s,// CDB:,," %s \
// RUN:    | %cdb -snul -lines -y . %t.exe >%t.out
// RUN: FileCheck %s < %t.out
module vector;

import core.simd;

// CDB: ld /f vector_cdb*
// enable case sensitive symbol lookup
// CDB: .symopt-1
// CDB: bp0 /1 `vector_cdb.d:92`
// CDB: g
// CHECK: Breakpoint 0 hit
// CDB: dv /t

int main()
{
    byte16  b16 = 1;
// CDB: ?? b16[1]
// CHECK: char 0n1
    ubyte16 ub16 = 2;
// CDB: ?? ub16[1]
// CHECK: unsigned char 0x02
    short8  s8 = 3;
// CDB: ?? s8[1]
// CHECK: short 0n3
    ushort8 us8 = 4;
// CDB: ?? us8[1]
// CHECK: unsigned short 4
    int4    i4 = 5;
// CDB: ?? i4[1]
// CHECK: int 0n5
    uint4   ui4 = 6;
// CDB: ?? ui4[1]
// CHECK: unsigned int 6
    long2   l2 = 7;
// CDB: ?? l2[1]
// CHECK: int64 0n7
    ulong2  ul2 = 8;
// CDB: ?? ul2[1]
// CHECK: unsigned int64 8
    float4  f4 = 9;
// CDB: ?? f4[1]
// CHECK: float 9
    double2 d2 = 10;
// CDB: ?? d2[1]
// CHECK: double 10
    void16  v16 = b16 + 10;
// CDB: ?? v16[1]
// v16 displayed as ubyte16
// CHECK: unsigned char 0x0b

    byte32  b32 = 12;
// CDB: ?? b32[1]
// CHECK: char 0n12
    ubyte32 ub32 = 13;
// CDB: ?? ub32[1]
// CHECK: unsigned char 0x0d
    short16 s16 = 14;
// CDB: ?? s16[1]
// CHECK: short 0n14
    ushort16 us16 = 15;
// CDB: ?? us16[1]
// CHECK: unsigned short 0xf
    int8    i8 = 16;
// CDB: ?? i8[1]
// CHECK: int 0n16
    uint8   ui8 = 17;
// CDB: ?? ui8[1]
// CHECK: unsigned int 0x11
    long4   l4 = 18;
// CDB: ?? l4[1]
// CHECK: int64 0n18
    ulong4  ul4 = 19;
// CDB: ?? ul4[1]
// CHECK: unsigned int64 0x13
    float8  f8 = 20;
// CDB: ?? f8[1]
// CHECK: float 20
    double4 d4 = 21;
// CDB: ?? d4[1]
// CHECK: double 21
    void32  v32 = b32 + 10;
// CDB: ?? v32[1]
// v32 displayed as ubyte32
// CHECK: unsigned char 0x16

    return 0; // BP
}
// CDB: q
// CHECK: quit:
