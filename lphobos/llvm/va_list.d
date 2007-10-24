module llvm.va_list;

alias void* va_list;

/*

version(X86)
{
    alias void* va_list;
}
else version(X86_64)
{
    struct X86_64_va_list
    {
        uint gp_offset;
        uint fp_offset;
        void* overflow_arg_area;
        void* reg_save_area;
    }
    alias X86_64_va_list va_list;
}
else
static assert("only x86 and x86-64 support va_list");

*/
