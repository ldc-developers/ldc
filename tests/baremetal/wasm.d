// Compile and link directly to naked WebAssembly.

// REQUIRES: target_WebAssembly
// REQUIRES: link_WebAssembly
// RUN: %ldc -mtriple=wasm32-unknown-unknown -w %s

extern(C): // no mangling

void _start() {}

double add(double a, double b) { return a + b; }
