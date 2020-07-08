// REQUIRES: atleast_llvm800, target_AVR

// RUN: %ldc -mtriple=avr -betterC -output-ll -of=%t.ll %s && FileCheck %s < %t.ll


// test predefined versions:

version (AVR) {} else static assert(0);
version (D_SoftFloat) {} else static assert(0);

// make sure TLS globals are emitted as regular __gshared globals:

// CHECK: @_D3avr13definedGlobali = global i32 123
int definedGlobal = 123;
// CHECK: @_D3avr14declaredGlobali = external global i32
extern int declaredGlobal;

// test address space of global ctor/dtor function pointers:

// CHECK: @llvm.global_ctors = appending global [1 x { i32, void () addrspace(1)*, i8* }]
// CHECK-SAME: @_D3avr4ctorFZv
pragma(crt_constructor)
void ctor() {}

// CHECK: @llvm.global_dtors = appending global [1 x { i32, void () addrspace(1)*, i8* }]
// CHECK-SAME: @_D3avr4dtorFZv
pragma(crt_destructor)
void dtor() {}
