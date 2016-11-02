// RUN: %ldc -gline-tables-only --output-ll -c %s -of%t.ll && FileCheck %s < %t.ll
// Checks that ldc with "-gline-tables-only" doesn't emit debug info
// for variables and types.

// CHECK-NOT: DW_TAG_variable
int global = 42;

// CHECK-NOT: DW_TAG_typedef
// CHECK-NOT: DW_TAG_const_type
// CHECK-NOT: DW_TAG_pointer_type
// CHECK-NOT: DW_TAG_array_type
alias constCharPtrArray = const(char)*[10];

// CHECK-NOT: DW_TAG_structure_type
struct S {
  // CHECK-NOT: DW_TAG_member
  char a;
  double b;
  constCharPtrArray c;
}

// CHECK-NOT: DW_TAG_enumerator
// CHECK-NOT: DW_TAG_enumeration_type
enum E { ZERO = 0, ONE = 1 }

// CHECK-NOT: DILocalVariable
int sum(int p, int q) {
  int r = p + q;
  S s;
  E e;
  return r;
}
