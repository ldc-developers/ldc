// A more complex wasm example using Phobos templates (=> -betterC to keep it simple).

// REQUIRES: target_WebAssembly
// REQUIRES: link_WebAssembly

// RUN: %ldc -mtriple=wasm32-unknown-unknown-wasm -betterC -w %s -of=%t.wasm

// make sure the .wasm file contains `myExportedFoo` (https://github.com/ldc-developers/ldc/issues/3023)
// RUN: grep myExportedFoo %t.wasm

extern(C):

void _start() {}

void __assert(const(char)* msg, const(char)* file, uint line) {}

export int myExportedFoo()
{
    import std.algorithm, std.range;
    auto range = 100.iota().stride(2).take(5);
    return range.sum();
}
