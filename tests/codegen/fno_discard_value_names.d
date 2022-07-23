// RUN: %ldc -c --print-after-all %s 2>&1 | FileCheck %s --check-prefix=DISCARD_CHECK
// RUN: %ldc -c --print-after-all --fno-discard-value-names %s 2>&1 | FileCheck %s --check-prefix=NO_DISCARD_CHECK

// DISCARD_CHECK-NOT: %first_argument_name
// DISCARD_CHECK-NOT: %a_second_argument
// NO_DISCARD_CHECK: %first_argument_name
// NO_DISCARD_CHECK: %a_second_argument

void foofoo(int first_argument_name, int a_second_argument) {}
