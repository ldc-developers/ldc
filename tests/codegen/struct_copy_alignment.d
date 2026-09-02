// Struct copies and initialisations carry the D type alignment on the memcpy/memset.
// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

struct S8 { long a; int b; }
align(1) struct P4 { int a; ubyte b; }
align(16) struct A16 { int a; }

// CHECK-LABEL: define {{.*}}_D{{.*}}copy8
void copy8(ref S8 dst, ref S8 src)
{
    // CHECK: call void @llvm.memcpy.{{.*}}(ptr align 8 %{{.*}}, ptr align 8 %{{.*}}, i{{32|64}} 16
    dst = src;
}

// CHECK-LABEL: define {{.*}}_D{{.*}}copy1
void copy1(ref P4 dst, ref P4 src)
{
    // CHECK: call void @llvm.memcpy.{{.*}}(ptr align 1 %{{.*}}, ptr align 1 %{{.*}}, i{{32|64}} 5
    dst = src;
}

// CHECK-LABEL: define {{.*}}_D{{.*}}copy16
void copy16(ref A16 dst, ref A16 src)
{
    // CHECK: call void @llvm.memcpy.{{.*}}(ptr align 16 %{{.*}}, ptr align 16 %{{.*}}, i{{32|64}} 16
    dst = src;
}

// CHECK-LABEL: define {{.*}}_D{{.*}}init8
void init8(ref S8 dst)
{
    // CHECK: call void @llvm.memcpy.{{.*}}(ptr align 8 %{{.*}}, ptr align 8 %{{.*}}, i{{32|64}} 16
    dst = S8.init;
}
