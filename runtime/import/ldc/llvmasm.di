module ldc.llvmasm;

struct __asmtuple_t(T...)
{
    T v;
}

pragma(llvm_inline_asm)
{
    void __asm( )(char[] asmcode, char[] constraints, ...);
    T    __asm(T)(char[] asmcode, char[] constraints, ...);

    template __asmtuple(T...)
    {
        __asmtuple_t!(T) __asmtuple(char[] asmcode, char[] constraints, ...);
    }
}
