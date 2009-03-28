module ldc.llvmasm;

pragma(llvm_inline_asm)
template __asm()
{
    void __asm(char[] asmcode, char[] constraints, ...);
}

pragma(llvm_inline_asm)
template __asm(T)
{
    T __asm(char[] asmcode, char[] constraints, ...);
}
