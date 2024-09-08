// REQUIRES: target_WebAssembly

// RUN: %ldc -mtriple=wasm32-unknown-emscripten -c %s


// test predefined versions:

version (Emscripten) {} else static assert(0);
version (linux) {} else static assert(0);
version (Posix) {} else static assert(0);
version (CRuntime_Musl) {} else static assert(0);
version (CppRuntime_LLVM) {} else static assert(0);


// verify that some druntime C bindings are importable:

import core.stdc.stdio;

extern(C) void _start() {
    puts("Hello world");
}
