void foo()
{
    version(X86)
    asm
    {
        naked;
        jmp pass;
        hlt;
pass:   ret;
    }
    else version(X86_64)
    asm
    {
        naked;
        jmp pass;
        hlt;
pass:   ret;
    }
    else static assert(0, "todo");
}

void main()
{
    foo();
}
