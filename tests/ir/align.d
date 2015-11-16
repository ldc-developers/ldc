// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

align(32) struct Outer { int a; }
struct Inner { align(32) int a; }

void main() {
  static Outer globalOuter;
  static Inner globalInner;
  // CHECK: constant %align.Outer_init zeroinitializer, align 32
  // CHECK: constant %align.Inner_init zeroinitializer, align 32

  Outer outer;
  Inner inner;
  // CHECK: %outer = alloca %align.Outer, align 32
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
}
