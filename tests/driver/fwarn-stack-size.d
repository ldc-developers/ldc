// Test --fwarn-stack-trace functionality

// REQUIRES: atleast_llvm1300

// RUN:     %ldc -c --fwarn-stack-size=2000 %s
// RUN:     %ldc -c --fwarn-stack-size=200  %s 2>&1 | FileCheck %s

// RUN: not %ldc -w -c --fwarn-stack-size=200  %s 2>&1 | FileCheck %s

module fwarnstacksize;

void small_stack()
{
    byte[100] a;
}

// CHECK: warning: stack frame size {{.*}} exceeds limit (200) in function {{.*}}14fwarnstacksize9big_stack
void big_stack()
{
    byte[1000] b;
}
