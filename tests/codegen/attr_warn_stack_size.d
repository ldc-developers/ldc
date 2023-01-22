// Test diagnostics for warnings on stack size

// REQUIRES: atleast_llvm1400

// RUN: %ldc -wi -c %s 2>&1 | FileCheck %s

import ldc.attributes;

// CHECK: {{.*}}Warning: stack frame size (8) exceeds limit (1) in '{{.*}}foobar{{.*}}'
@llvmAttr("warn-stack-size", "1")
float foobar(int f)
{
    float _ = f;
    return _;
}
