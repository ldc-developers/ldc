int foo()
{
    version(X86)
    asm { mov EAX, 42; }
    else static assert(0, "todo");
}

ulong bar()
{
    version(X86)
    asm { mov EAX, 0xFF; mov EDX, 0xAA; }
    else static assert(0, "todo");
}

void main()
{
    long l = 1;
    l = 2;
    l = 4;
    l = 8;
    assert(foo() == 42);
    assert(bar() == 0x000000AA000000FF);
}
