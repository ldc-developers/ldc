/**
 * This module implements the runtime-part of LDC exceptions
 * on WebAssembly (Wasm EH)
 */

module ldc.eh_wasm;

version (WASI):

extern (C) void _d_throw_exception(Throwable t) {
    import core.stdc.stdio : fwrite, stdout, putc;
    import ldc.intrinsics : llvm_trap;

    auto msg = t.toString();
    fwrite(msg.ptr, msg.length, 1, stdout);
    putc('\n', stdout);

    llvm_trap();
}

extern (C) void _Unwind_Resume(void*) {
    import core.stdc.stdio : puts;
    import ldc.intrinsics : llvm_trap;

    puts("Cannot EH unwind on Wasm (yet).");
    llvm_trap();
}

extern (C) void _d_eh_enter_catch(void*) {
    import core.stdc.stdio : puts;
    import ldc.intrinsics : llvm_trap;

    puts("Cannot EH catch on Wasm (yet).");
    llvm_trap();
}
