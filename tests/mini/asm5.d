int foo()
{
    version(X86)
    {
      asm { mov EAX, 42; }
    } else version(X86_64)
    {
      asm { movq RAX, 42; }
    }
    else static assert(0, "todo");
}

ulong bar()
{
    version(X86)
    {
      asm { mov EAX, 0xFF; mov EDX, 0xAA; }
    } else version(X86_64)
    {
      asm { movq RAX, 0xFF; }
    }
    else static assert(0, "todo");
}

void main()
{
    long l = 1;
    l = 2;
    l = 4;
    l = 8;
    assert(foo() == 42);
    version(X86)
    {
        assert(bar() == 0x000000AA000000FF);
    } else version(X86_64)
    {
        assert(bar() == 0x00000000000000FF);
    }
}
