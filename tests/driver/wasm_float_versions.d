// REQUIRES: target_WebAssembly

// RUN: %ldc -c -o- %s -mtriple=wasm32-unknown-unknown-wasm

version (D_HardFloat) {} else static assert(0);
version (D_SoftFloat) static assert(0);
