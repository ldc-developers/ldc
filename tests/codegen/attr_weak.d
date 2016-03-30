// Test linking+running a program with @weak functions

// RUN: %ldc -O3 %S/inputs/attr_weak_input.d -c -of=%t%obj
// RUN: %ldc -O3 %t%obj -run %s


import ldc.attributes;

extern(C) int return_two() {
  return 2;
}

// Should be overridden by attr_weak_input.d
extern(C) @weak int return_seven() {
  return 1;
}

void main() {
  assert( return_two() == 2 );
  assert( return_seven() == 7 );
}
