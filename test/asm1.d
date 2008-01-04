module asm1;

void main()
{
    version(LLVM_InlineAsm_X86_64)
    {
        long x;
        asm
        {
            mov RAX, 42L;
            mov x, RAX;
        }
        printf("x = %ld\n", x);
    }
    else
    {
        static assert(0, "no llvm inline asm for this platform yet");
    }
}
