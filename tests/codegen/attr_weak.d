// Test linking+running a program with @weak functions

// RUN: %ldc -O3 %S/inputs/attr_weak_input.d -c -of=%t-dir/attr_weak_input%obj
// RUN: %ldc -O3 %s %t-dir/attr_weak_input%obj -of=%t%exe
// RUN: %t%exe


import ldc.attributes;

// should take precedence over and not conflict with weak attr_weak_input.return_two
extern(C) int return_two() {
    return 123;
}

// should be overridden by strong attr_weak_input.return_seven
extern(C) @weak int return_seven() {
  return 456;
}

void main() {
  assert( return_two() == 123 );
  assert( return_seven() == 7 );
}
