// Compile and link directly to WebAssembly.

// REQUIRES: target_WebAssembly
// RUN: %ldc -mtriple=wasm32-unknown-unknown-wasm %s %baremetal_args

extern(C): // no mangling, no arguments order reversal

void _start() {}

double add(double a, double b) { return a + b; }
