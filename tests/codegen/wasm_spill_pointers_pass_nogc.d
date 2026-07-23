// REQUIRES: target_WebAssembly, link_WebAssembly

// optimize to create SSA IR
// RUN: %ldc -mtriple=wasm32-unknown-unknown -betterC -O3 -c -output-ll -of=%t.ll %s
// RUN: FileCheck %s < %t.ll

// Make sure nogc compilations (e.g. -betterC) don't run the
// spill pointers pass

// CHECK-NOT: stackSpill

void* getPtr();
void blackbox();
void* test()
{

    auto ptr = getPtr();
    blackbox();
    return ptr;
}
