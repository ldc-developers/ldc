// RUN: not %ldc -c %s 2>&1 | FileCheck %s

// CHECK: gh4938.d(4): Error: expression `&"whoops"w[0]` is not a constant
immutable(wchar)* x = &"whoops"w[0];
