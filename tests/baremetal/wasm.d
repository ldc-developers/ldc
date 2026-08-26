// Compile and link directly to naked WebAssembly.

// REQUIRES: target_WebAssembly
// REQUIRES: internal_lld
// RUN: %ldc -mtriple=wasm32-unknown-unknown -w %s

extern(C): // no mangling

void _start() {}

double add(double a, double b) { return a + b; }
