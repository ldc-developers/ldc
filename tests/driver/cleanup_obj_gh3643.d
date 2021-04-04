module tests.driver.cleanup_obj;

// REQUIRES: target_X86
// RUN: %ldc -cleanup-obj -oq -gcc=echo -mtriple=x86_64-linux %s | FileCheck %s

// CHECK: objtmp-ldc-{{([a-zA-Z0-9]{6})[/\\]}}tests.driver.cleanup_obj.o
