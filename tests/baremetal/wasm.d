// Compile and link directly to WebAssembly.

// REQUIRES: target_WebAssembly
// REQUIRES: internal_lld
// RUN: %ldc -mtriple=wasm32-unknown-unknown-wasm -w -link-internally %s %baremetal_args

extern(C): // no mangling, no arguments order reversal

void _start() {}

double add(double a, double b) { return a + b; }
