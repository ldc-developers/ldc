module ldc.gcroot_wasm;

version (WebAssembly):

/*
On WebAssembly, a working conservative GC is blocked on being able to see
pointers that might be held somewhere outside of Wasm's linear memory.

Wasm has both an infinite set of "locals" (~registers), AND it's own
value stack (Wasm is a stack machine). Neither of these are easily
introspectable from arbitrary code. So unlike native platforms, we can't just
spill all the "registers" onto the in-memory stack and be done with it.

Furthermore, due largely to `union`, there may be pointers smuggled in types
other than the appropriate `i32` on wasm32 or `i64` on wasm64.
E.g. `union U { double d; void* p }` holds a pointer in what LLVM sees
as just a double (and becomes Wasm f64).

To work around these limitations, we use a dummy "runtime" hook for Wasm,
with an IR signature of:
`declare void @_d_stack_gcroot(ptr captures(read_provenence)) memory(inaccessiblemem: write);`

Whenever an `alloca` is created for a D type that contains pointers,
a call to `_d_stack_gcroot` is inserted passing the `alloca` ptr in.
This inhibits SROA and mem2reg, forcing the value to remain on the stack.
Rather than spilling things back, we simply force them to stay.

`captures(read_provenence)` makes sure that any stores are visible before calls
(which LLVM will assume might read the pointer), but avoids forcing reload
after such calls (it is assumed the D does not use a moving GC,
so we can avoid this pessimization; e.g. store forwarding optimizations
still work across calls).

These calls are not inserted on @nogc functions (and in nogc compilations, e.g. betterC).
This relies on the transitivity of @nogc. If a function is @nogc, it may not
call non-@nogc functions, which we assume to mean GC.collect will never be invoked.

A pass is used remove _d_stack_gcroot calls after the optimizations are done.
This means running non-LDC opt on the results of --output-ll will break the GC.
*/

// IMPORTANT: @nogc to prevent recursion (it calling _d_stack_gcroot on its parameter)
extern (C) void _d_stack_gcroot(void*) @nogc {}
