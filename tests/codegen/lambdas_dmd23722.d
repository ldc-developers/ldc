// Test that colliding lambda mangles don't lead to symbol collision during linking,
// see https://issues.dlang.org/show_bug.cgi?id=23722

// compile both modules separately, then link and check runtime output
// RUN: %ldc -c %S/inputs/lambdas_dmd23722b.d -of=%t_b%obj
// RUN: %ldc -I%S/inputs %s %t_b%obj -of=%t%exe
// RUN: %t%exe | FileCheck %s

import lambdas_dmd23722b;

// do_y should call A.y (and print "y")
void do_y() {
    A.y();
}

void main() {
    // CHECK: y
    do_y(); // should print y
    // CHECK-NEXT: x
    do_x(); // should print x
}
