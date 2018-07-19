// A more complex wasm example using Phobos templates (=> -betterC to keep it simple).

// REQUIRES: target_WebAssembly
// RUN: %ldc -mtriple=wasm32-unknown-unknown-wasm -link-internally -betterC %s

extern(C):

void _start() {}

void __assert(const(char)* msg, const(char)* file, uint line) {}

int foo()
{
    import std.algorithm, std.range;
    auto range = 100.iota().stride(2).take(5);
    return range.sum();
}
