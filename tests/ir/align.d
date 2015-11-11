// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s --check-prefix LLVM < %t.ll

align(32) struct Outer { int a; }
struct Inner { align(32) int a; }

void main() {
  static Outer globalOuter;
  static Inner globalInner;
  // LLVM: constant %align.Outer_init zeroinitializer, align 32
  // LLVM: constant %align.Inner_init zeroinitializer, align 32

  Outer outer;
  Inner inner;
  // LLVM: %outer = alloca %align.Outer, align 32
  // LLVM: %inner = alloca %align.Inner, align 32
}
