module ldc.gcroot_wasm;

version (WebAssembly):

// does nothing; just used as an (de)optimizer hint
// calls to this should normally be eliminated by the StripStackGCRoot pass.
//
// IMPORTANT: @nogc to prevent recursion (it calling _d_stack_gcroot on its parameter)
extern (C) void _d_stack_gcroot(void*) @nogc {}
