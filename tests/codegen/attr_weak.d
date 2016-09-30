// Test linking+running a program with @weak function

// RUN: %ldc -O3 %S/inputs/attr_weak_input.d -c -of=%T/attr_weak_input%obj
// RUN: %ldc -O3 %T/attr_weak_input%obj %s -of=%t%exe
// RUN: %t%exe


import ldc.attributes;

// Should be overridden by attr_weak_input.d (but only because its object
// file is specified before this one for the linker).
// The @weak attribute prevents the optimizer from making any assumptions
// though, so the call below is not inlined.
extern(C) @weak int return_seven() {
  return 1;
}

void main() {
  assert( return_seven() == 7 );
}
