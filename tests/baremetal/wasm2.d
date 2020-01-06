// A -betterC wasm example.

// REQUIRES: target_WebAssembly
// REQUIRES: link_WebAssembly

// RUN: %ldc -mtriple=wasm32-unknown-unknown-wasm -betterC -w %s -of=%t.wasm
// RUN: %ldc -mtriple=wasm32-unknown-unknown-wasm -betterC -w -fvisibility=hidden %s -of=%t_hidden.wasm

// make sure the .wasm files contain `myExportedFoo` (https://github.com/ldc-developers/ldc/issues/3023)
// RUN: grep myExportedFoo %t.wasm
// RUN: grep myExportedFoo %t_hidden.wasm

extern(C):

void _start() {}

void __assert(const(char)* msg, const(char)* file, uint line) {}

export void myExportedFoo(double x)
{
    assert(x > 0);
}
