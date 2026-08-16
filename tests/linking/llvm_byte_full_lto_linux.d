// Full LTO + b8 partner (hand-written IR), native AArch64 Linux link.
// See docs/byteType.md layer 3. Partner: inputs/llvm_byte_lto_partner_aarch64_linux.ll

// REQUIRES: LTO && internal_lld && atleast_llvm23 && llvm_ir_b8 && target_AArch64 && host_AArch64 && Linux

// UNSUPPORTED: Windows

// RUN: %llvm-as %S/inputs/llvm_byte_lto_partner_aarch64_linux.ll -o %t_p.bc
// RUN: %ldc -mtriple=aarch64-unknown-linux-gnu -fc-interop-llvm-byte -flto=full -O1 -link-internally %t_p.bc %s -of=%t_x%exe
// RUN: test -f %t_x%exe

extern (C) ubyte llvm_byte_lto_add_one(ubyte x);
extern (C) void llvm_byte_lto_sink_uchar(ubyte x);

void main() {
    llvm_byte_lto_sink_uchar(0);
    assert(llvm_byte_lto_add_one(5) == 6);
}
