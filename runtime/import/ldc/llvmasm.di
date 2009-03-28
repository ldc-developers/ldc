module ldc.llvmasm;

pragma(llvm_inline_asm)
template __asm()
{
    void __asm(char[] asmcode, char[] constraints, ...);
}
