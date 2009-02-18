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
      asm { movq RAX, 0xAA000000FF; }
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
    assert(bar() == 0xAA000000FF);
}
