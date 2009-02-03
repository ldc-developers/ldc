extern(C) int printf(char*, ...);

void main()
{
    int i = func();
    printf("%d\n", i);
    assert(i == 42);
}

int func()
{
    version (LLVM_InlineAsm_X86)
    {
        asm
        {
            naked;
            mov EAX, 42;
            ret;
        }
    }
    else version(LLVM_InlineAsm_X86_64)
    {
        asm
        {
            naked;
            movq RAX, 42;
            ret;
        }
    }
}
