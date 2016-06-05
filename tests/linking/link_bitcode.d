// Test linking with an LLVM bitcode file

// RUN: %ldc -c -output-bc -I%S %S/inputs/link_bitcode_input.d -of=%t.bc
// RUN: %ldc -c -output-bc -I%S %S/inputs/link_bitcode_import.d -of=%t2.bc
// RUN: %ldc -c -output-bc -I%S %S/inputs/link_bitcode_input3.d -of=%t3.bc
// RUN: %ldc -c -singleobj -output-bc %t.bc %t3.bc %s
// RUN: %ldc -c -singleobj -I%S %t.bc %s %S/inputs/link_bitcode_input3.d
// RUN: %ldc -c -singleobj -I%S %t.bc %S/inputs/link_bitcode_input3.d %s
// RUN: %ldc -c -I%S %t.bc %S/inputs/link_bitcode_input3.d %s

// RUN: %ldc %t.bc %t2.bc %t3.bc -run %s
// RUN: %ldc %t.bc %S/inputs/link_bitcode_import.d %t3.bc -run %s

// Defined in input/link_bitcode_input.d
extern(C) int return_seven();

void main() {
  assert( return_seven() == 7 );
}
