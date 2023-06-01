// REQUIRES: ASan

// RUN: %ldc -femit-local-var-lifetime -g -fsanitize=address %s -of=%t%exe
// RUN: not %t%exe 2>&1 | FileCheck %s

// CHECK: ERROR: AddressSanitizer: stack-use-after-scope
// CHECK-NEXT: WRITE of size 4

void useAfterScope() {
  int* p;
  {
    int x = 0;
    p = &x;  // cannot statically disallow this because
    *p = 1;  // this is a valid use of things
  }
// CHECK-NEXT: #0 {{.*}} in {{.*}}asan_use_after_scope.d:[[@LINE+1]]
  *p = 5;  // but then this can happen... stack use after scope bug!
}

void main()
{
// CHECK-NEXT: #1 {{.*}} in {{.*}}asan_use_after_scope.d:[[@LINE+1]]
    useAfterScope();
}
