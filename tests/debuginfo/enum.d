// RUN: %ldc -g -output-ll -of=%t.ll %s
// RUN: FileCheck %s < %t.ll

enum Foo : float
{
  Bar = 3.15,
  Bar2,
  Bar3
}

void openBar (Foo f) { asm { int 3; } }

void main()
{
  openBar(Foo.Bar3);
}

// CHECK-NOT: DW_TAG_enumeration_type
