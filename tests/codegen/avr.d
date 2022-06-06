// REQUIRES: target_AVR

// RUN: %ldc -mtriple=avr -betterC -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

version (AVR) {} else static assert(0);
version (D_SoftFloat) {} else static assert(0);

// make sure TLS globals are emitted as regular __gshared globals:

// CHECK: @_D3avr13definedGlobali = global i32 123
int definedGlobal = 123;
// CHECK: @_D3avr14declaredGlobali = external global i32
extern int declaredGlobal;
