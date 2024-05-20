// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// Fails on Windows_x86, see https://github.com/ldc-developers/ldc/issues/1356
// XFAIL: Windows_x86

align(32) struct Outer { int a = 1; }
// CHECK-DAG: _D5align5Outer6__initZ = constant %align.Outer {{.*}}, align 32
struct Inner { align(32) int a = 1; }
// CHECK-DAG: _D5align5Inner6__initZ = constant %align.Inner {{.*}}, align 32

align(1) ubyte globalByte1;
// CHECK-DAG: _D5align11globalByte1h = {{.*}} align 1
static Outer globalOuter;
// CHECK-DAG: _D5align11globalOuterSQu5Outer = {{.*}} align 32
static Inner globalInner;
// CHECK-DAG: _D5align11globalInnerSQu5Inner = {{.*}} align 32

Outer passAndReturnOuterByVal(Outer arg) { return arg; }
// CHECK: define{{.*}} void @{{.*}}_D5align23passAndReturnOuterByValFSQBh5OuterZQl
/* the 32-bit x86 ABI substitutes the sret attribute by inreg */
// CHECK-SAME: ptr {{noalias sret.*|inreg noalias}} align 32 %.sret_arg
/* How the arg is passed by value is ABI-specific, but the pointer must be aligned.
 * When the argument is passed as a byte array and copied into a stack alloc, that stack alloca must be aligned. */
// CHECK: {{(align 32 %arg|%arg = alloca %align.Outer, align 32)}}

Inner passAndReturnInnerByVal(Inner arg) { return arg; }
// CHECK: define{{.*}} void @{{.*}}_D5align23passAndReturnInnerByValFSQBh5InnerZQl
// CHECK-SAME: ptr {{noalias sret.*|inreg noalias}} align 32 %.sret_arg
// CHECK: {{(align 32 %arg|%arg = alloca %align.Inner, align 32)}}

void main() {
  Outer outer;
  // CHECK: %outer = alloca %align.Outer, align 32
  Inner inner;
  // CHECK: %inner = alloca %align.Inner, align 32

  align(1) byte byte1;
  // CHECK: %byte1 = alloca i8, align 1
  align(16) byte byte16;
  // CHECK: %byte16 = alloca i8, align 16
  align(64) Outer outer64;
  // CHECK: %outer64 = alloca %align.Outer, align 64
  align(128) Inner inner128;
  // CHECK: %inner128 = alloca %align.Inner, align 128

  alias Byte8 = align(8) byte;
  Byte8 byte8;
  // Can aliases contain align(x) ?
  // C HECK: %byte8 = alloca i8, align 8
  // C HECK: %byte8 = alloca i8, align 1

  align(16) Outer outeroverride;
  // Yet undecided if align() should override type alignment:
  // C HECK: %outeroverride = alloca %align.Outer, align 16
  // C HECK: %outeroverride = alloca %align.Outer, align 32

  // CHECK: %.sret_tmp{{.*}} = alloca %align.Outer, align 32
  // CHECK: %.sret_tmp{{.*}} = alloca %align.Inner, align 32

  outer = passAndReturnOuterByVal(outer);
  // CHECK: call{{.*}} void @{{.*}}_D5align23passAndReturnOuterByValFSQBh5OuterZQl
  // CHECK-SAME: ptr {{noalias sret.*|inreg noalias}} align 32 %.sret_tmp
  // The argument is either passed by aligned (optimizer hint) pointer or as an array of i32/64 and copied into an aligned stack slot inside the callee.
  // CHECK-SAME: {{(align 32 %|\[[0-9]+ x i..\])}}

  inner = passAndReturnInnerByVal(inner);
  // CHECK: call{{.*}} void @{{.*}}_D5align23passAndReturnInnerByValFSQBh5InnerZQl
  // CHECK-SAME: ptr {{noalias sret.*|inreg noalias}} align 32 %.sret_tmp
  // The argument is either passed by aligned (optimizer hint) pointer or as an array of i32/64 and copied into an aligned stack slot inside the callee.
  // CHECK-SAME: {{(align 32 %|\[[0-9]+ x i..\])}}
}
