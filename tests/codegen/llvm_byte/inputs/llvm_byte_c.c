/* Input for llvm_byte_link_runtime.d: C side of extern(C) ubyte interop.
 * Built with the host C compiler as a relocatable object, then linked with LDC
 * using -fc-interop-llvm-byte (LLVM 23+ AArch64).
 */
unsigned char llvm_byte_c_add_uchar(unsigned char a, unsigned char b) {
  return (unsigned char)(a + b);
}

unsigned char llvm_byte_c_inc_uchar(unsigned char x) {
  return (unsigned char)(x + 1);
}
