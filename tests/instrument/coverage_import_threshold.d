// REQUIRES: Linux
// RUN: mkdir -p %t && cp %S/inputs/coverage_threshold_library.d %t
// RUN: cd %t && %ldc -cov -c coverage_threshold_library.d -of=a.o
// RUN: cd %t && %ldc -cov=100 -I. -c %s -of=b.o
// RUN: cd %t && %ldc a.o b.o -of=program && ./program
// RUN: cd %t && %ldc -cov=100 -c coverage_threshold_library.d -of=a.o
// RUN: cd %t && %ldc a.o b.o -of=program
// RUN: cd %t && not ./program 2>&1 | FileCheck %s
// CHECK: coverage_threshold_library.d is 50% covered, less than required 100%

import coverage_threshold_library : identity;

void main()
{
    assert(identity(3) == 3);
}
