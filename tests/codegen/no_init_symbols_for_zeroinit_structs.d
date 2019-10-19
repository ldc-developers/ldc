// RUN: %ldc -output-ll -of=%t.ll %s
// RUN: FileCheck %s < %t.ll

module mod;

// CHECK-NOT: _D3mod5Empty6__initZ
struct Empty {}

// CHECK-NOT: _D3mod7WithInt6__initZ
struct WithInt { int a; }

// CHECK-NOT: _D3mod13WithZeroFloat6__initZ
struct WithZeroFloat { float a = 0; }
