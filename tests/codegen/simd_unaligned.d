// Tests unaligned load and stores of SIMD types

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -run %s

import core.simd;
import ldc.simd;

// CHECK-LABEL: define{{.*}} @{{.*}}loads
void loads(void *p)
{
    // CHECK: load <4 x float>{{.*}} align 1
    loadUnaligned!float4(cast(float*)p);

    const float[4] f4buf = void;
    immutable double[2] f8buf = void;
    ubyte[16] u1buf = void;
    ushort[8] u2buf = void;
    uint[4] u4buf = void;
    ulong[2] u8buf = void;
    byte[16] i1buf = void;
    short[8] i2buf = void;
    int[4] i4buf = void;
    long[2] i8buf = void;
    // CHECK: load <4 x float>{{.*}} align 1
    loadUnaligned!float4(f4buf.ptr);
    // CHECK: load <2 x double>{{.*}} align 1
    loadUnaligned!double2(f8buf.ptr);
    // CHECK: load <16 x i8>{{.*}} align 1
    loadUnaligned!ubyte16(u1buf.ptr);
    // CHECK: load <8 x i16>{{.*}} align 1
    loadUnaligned!ushort8(u2buf.ptr);
    // CHECK: load <4 x i32>{{.*}} align 1
    loadUnaligned!uint4(u4buf.ptr);
    // CHECK: load <2 x i64>{{.*}} align 1
    loadUnaligned!ulong2(u8buf.ptr);
    // CHECK: load <16 x i8>{{.*}} align 1
    loadUnaligned!byte16(i1buf.ptr);
    // CHECK: load <8 x i16>{{.*}} align 1
    loadUnaligned!short8(i2buf.ptr);
    // CHECK: load <4 x i32>{{.*}} align 1
    loadUnaligned!int4(i4buf.ptr);
    // CHECK: load <2 x i64>{{.*}} align 1
    loadUnaligned!long2(i8buf.ptr);
}

// CHECK-LABEL: define{{.*}} @{{.*}}stores
void stores(void *p)
{
    float8 f8 = void;
    int8 i8 = void;
    // CHECK: store <8 x float>{{.*}} align 1
    storeUnaligned!float8(f8, cast(float*)p);
    // CHECK: store <8 x i32>{{.*}} align 1
    storeUnaligned!int8(i8, cast(int*)p);
}

void checkStore(int *a)
{
    immutable int4 v = [0, 10, 20, 30];
    // CHECK: store <4 x i32>{{.*}} align 1
    storeUnaligned!int4(v, a);
    assert(v.array == a[0..4]);
}

void main()
{
    loads(getMisalignedPtr());
    stores(getMisalignedPtr());
    checkStore(cast(int*)getMisalignedPtr());
}

import ldc.attributes;
align(32) char[100] dummy = void;
void* getMisalignedPtr()
@weak // disallows reasoning and inlining of this function
{
    return &dummy[1];
};
