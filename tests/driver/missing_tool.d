// Test that LDC complains about a missing tool.

// RUN: not %ldc %s -gcc=IdontExist -linker=IdontExist 2>&1 | FileCheck %s
// CHECK: Error: cannot find program `IdontExist`

void main() {}
