module tests.codegen.llvm_byte.llvm_byte_link_runtime;

// Runtime test: D with -fc-interop-llvm-byte links against a C object and runs.
// Requires AArch64 host so the default target matches the b8 AArch64 ABI path
// and -run executes a native binary (see docs/byteType.md staged tests).

// REQUIRES: atleast_llvm23 && llvm_ir_b8 && target_AArch64 && host_AArch64

// Host C compiler; skip Windows where `cc` is not in the lit environment.
// UNSUPPORTED: Windows

// RUN: cc -c -o %t_c.o %S/inputs/llvm_byte_c.c
// RUN: %ldc -fc-interop-llvm-byte %t_c.o %s -run

extern (C) ubyte llvm_byte_c_add_uchar(ubyte a, ubyte b);
extern (C) ubyte llvm_byte_c_inc_uchar(ubyte x);

void main() {
    assert(llvm_byte_c_add_uchar(3, 40) == 43);
    assert(llvm_byte_c_inc_uchar(cast(ubyte) 41) == 42);
}
