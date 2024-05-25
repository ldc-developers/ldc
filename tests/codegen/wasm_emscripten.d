// REQUIRES: target_WebAssembly

// RUN: %ldc -mtriple=wasm32-unknown-emscripten -w -output-ll -of=%t.ll %s
// RUN: FileCheck %s < %t.ll

// test predefined versions:
version (Emscripten) {} else static assert(0);

void foo() {}

// CHECK: target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-f128:64-n32:64-S128-ni:1:10:20"
