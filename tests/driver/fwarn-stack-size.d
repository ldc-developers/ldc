// Test --fwarn-stack-trace functionality

// RUN:     %ldc -c --fwarn-stack-size=2000 %s
// RUN:     %ldc -c --fwarn-stack-size=200  %s 2>&1 | FileCheck %s

// RUN: not %ldc -w -c --fwarn-stack-size=200  %s 2>&1 | FileCheck %s

// Test that IR caching does not hide the warning-error in a second compilation run
// RUN: not %ldc -cache=%t-dir -w -c --fwarn-stack-size=200 %s 2>&1 | FileCheck %s
// RUN: not %ldc -cache=%t-dir -w -c --fwarn-stack-size=200 %s 2>&1 | FileCheck %s
// Test that indeed the IR cache does not exist
// RUN: not %prunecache -f %t-dir --max-bytes=1

module fwarnstacksize;

void small_stack()
{
    byte[100] a;
}

// CHECK: warning: {{(<unknown>:0:0: )?}}stack frame size {{.*}} exceeds limit (200) in function {{.*}}14fwarnstacksize9big_stack
void big_stack()
{
    byte[1000] b;
}
