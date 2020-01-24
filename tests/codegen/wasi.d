// REQUIRES: atleast_llvm800, target_WebAssembly

// RUN: %ldc -mtriple=wasm32-unknown-wasi -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

version (WASI) {} else static assert(0);
version (CRuntime_WASI) {} else static assert(0);

// make sure TLS globals are emitted as regular __gshared globals:

// CHECK: @_D4wasi13definedGlobali = global i32 123
int definedGlobal = 123;
// CHECK: @_D4wasi14declaredGlobali = external global i32
extern int declaredGlobal;
