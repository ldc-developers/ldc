// Struct copies and zero-initialisations through pointers and refs carry the
// pointee type's alignment.
// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

struct S8 { long a; int b; }
align(1) struct P5 { int a; ubyte b; }
align(16) struct A16 { int a; }
struct Z8 { long a; long b; }
struct I8 { long a = 1; int b; }
align(1) struct Packed { ubyte pad; S8 s; }
class C { S8 s; }

// CHECK-LABEL: define {{.*}}_D{{.*}}refCopy8
void refCopy8(ref S8 dst, ref S8 src)
{
    // CHECK: call void @llvm.memcpy.{{.*}}(ptr align 8 %{{.*}}, ptr align 8 %{{.*}}, i{{32|64}} 16
    dst = src;
}

// CHECK-LABEL: define {{.*}}_D{{.*}}refCopy1
void refCopy1(ref P5 dst, ref P5 src)
{
    // CHECK: call void @llvm.memcpy.{{.*}}(ptr align 1 %{{.*}}, ptr align 1 %{{.*}}, i{{32|64}} 5
    dst = src;
}

// CHECK-LABEL: define {{.*}}_D{{.*}}refCopy16
void refCopy16(ref A16 dst, ref A16 src)
{
    // CHECK: call void @llvm.memcpy.{{.*}}(ptr align 16 %{{.*}}, ptr align 16 %{{.*}}, i{{32|64}} 16
    dst = src;
}

// CHECK-LABEL: define {{.*}}_D{{.*}}derefCopy
void derefCopy(S8* dst, S8* src)
{
    // CHECK: call void @llvm.memcpy.{{.*}}(ptr align 8 %{{.*}}, ptr align 8 %{{.*}}, i{{32|64}} 16
    *dst = *src;
}

// CHECK-LABEL: define {{.*}}_D{{.*}}indexCopy
void indexCopy(S8* p, S8[] a, size_t i)
{
    // CHECK: call void @llvm.memcpy.{{.*}}(ptr align 8 %{{.*}}, ptr align 8 %{{.*}}, i{{32|64}} 16
    p[i] = a[i];
}

ref S8 refReturn(S8* p) { return *p; }

// CHECK-LABEL: define {{.*}}_D{{.*}}refReturnCopy
void refReturnCopy(S8* p, ref S8 src)
{
    // CHECK: call void @llvm.memcpy.{{.*}}(ptr align 8 %{{.*}}, ptr align 8 %{{.*}}, i{{32|64}} 16
    refReturn(p) = src;
}

// CHECK-LABEL: define {{.*}}_D{{.*}}refLocalCopy
void refLocalCopy(S8* p, ref S8 src)
{
    ref S8 r = *p;
    // CHECK: call void @llvm.memcpy.{{.*}}(ptr align 8 %{{.*}}, ptr align 8 %{{.*}}, i{{32|64}} 16
    r = src;
}

struct W
{
    S8 s;
    // CHECK-LABEL: define {{.*}}_D{{.*}}1W8copyFrom
    void copyFrom(ref W other)
    {
        // CHECK: call void @llvm.memcpy.{{.*}}(ptr align 8 %{{.*}}, ptr align 8 %{{.*}}, i{{32|64}} 16
        this = other;
    }
}

// CHECK-LABEL: define {{.*}}_D{{.*}}zeroInit
void zeroInit(out Z8 dst)
{
    // CHECK: call void @llvm.memset.{{.*}}(ptr align 8 %{{.*}}, i8 0, i{{32|64}} 16
}

// CHECK-LABEL: define {{.*}}_D{{.*}}staticInit
void staticInit(out I8 dst)
{
    // CHECK: call void @llvm.memcpy.{{.*}}(ptr align 8 %{{.*}}, ptr align 8 @{{.*}}2I86__initZ, i{{32|64}} 16
}

// Field addresses don't carry an alignment (yet).

// CHECK-LABEL: define {{.*}}_D{{.*}}packedFieldCopy
void packedFieldCopy(ref Packed dst, ref S8 src)
{
    // CHECK: call void @llvm.memcpy.{{.*}}(ptr align 1 %{{.*}}, ptr align 8 %{{.*}}, i{{32|64}} 16
    dst.s = src;
}

// CHECK-LABEL: define {{.*}}_D{{.*}}classFieldCopy
void classFieldCopy(C dst, ref S8 src)
{
    // CHECK: call void @llvm.memcpy.{{.*}}(ptr align 1 %{{.*}}, ptr align 8 %{{.*}}, i{{32|64}} 16
    dst.s = src;
}
