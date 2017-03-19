// Just make sure LDC doesn't necessarily enforce the .ll extension (issue #1843).
// RUN: %ldc -output-ll -of=%t.myIR %s && FileCheck %s < %t.myIR

// CHECK: define{{.*}} void @{{.*}}_D6gh18433fooFZv
void foo() {}
