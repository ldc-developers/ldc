// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

align(32) struct Outer { int a; }
struct Inner { align(32) int a; }

static Outer globalOuter;
// CHECK: constant %align.Outer_init zeroinitializer{{(, comdat)?}}, align 32
static Inner globalInner;
// CHECK: constant %align.Inner_init zeroinitializer{{(, comdat)?}}, align 32

Outer passAndReturnOuterByVal(Outer arg) { return arg; }
// CHECK: define{{.*}} void @{{.*}}_D5align23passAndReturnOuterByValFS5align5OuterZS5align5Outer
/* the 32-bit x86 ABI substitutes the sret attribute by inreg */
// CHECK-SAME: %align.Outer* {{noalias sret|inreg noalias}} align 32 %.sret_arg
/* how the arg is passed by value is ABI-specific, but the pointer must be aligned */
// CHECK-SAME: align 32 %

Inner passAndReturnInnerByVal(Inner arg) { return arg; }
// CHECK: define{{.*}} void @{{.*}}_D5align23passAndReturnInnerByValFS5align5InnerZS5align5Inner
// CHECK-SAME: %align.Inner* {{noalias sret|inreg noalias}} align 32 %.sret_arg
// CHECK-SAME: align 32 %

void main() {
  Outer outer;
  // CHECK: %outer = alloca %align.Outer, align 32
  Inner inner;
  // CHECK: %inner = alloca %align.Inner, align 32

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

  // CHECK: %.rettmp{{.*}} = alloca %align.Outer, align 32
  // CHECK: %.rettmp{{.*}} = alloca %align.Inner, align 32

  outer = passAndReturnOuterByVal(outer);
  // CHECK: call{{.*}} void @{{.*}}_D5align23passAndReturnOuterByValFS5align5OuterZS5align5Outer
  // CHECK-SAME: %align.Outer* {{noalias sret|inreg noalias}} align 32 %.rettmp
  // CHECK-SAME: align 32 %

  inner = passAndReturnInnerByVal(inner);
  // CHECK: call{{.*}} void @{{.*}}_D5align23passAndReturnInnerByValFS5align5InnerZS5align5Inner
  // CHECK-SAME: %align.Inner* {{noalias sret|inreg noalias}} align 32 %.rettmp1
  // CHECK-SAME: align 32 %
}
