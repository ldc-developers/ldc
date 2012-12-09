module ldc.llvmasm;

struct __asmtuple_t(T...)
{
    T v;
}

pragma(LDC_inline_asm)
{
    void __asm()(const(char)[] asmcode, const(char)[] constraints, ...);
    T __asm(T)(const(char)[] asmcode, const(char)[] constraints, ...);

    template __asmtuple(T...)
    {
        __asmtuple_t!(T) __asmtuple(const(char)[] asmcode, const(char)[] constraints, ...);
    }
}
