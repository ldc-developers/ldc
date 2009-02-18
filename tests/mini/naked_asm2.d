int foo()
{
    static size_t fourty2 = 42;
    version(X86)
    asm
    {
        naked;
        mov EAX, fourty2;
        ret;
    }
    else version (X86_64)
    {
      asm
      {
	naked;
	movq RAX,fourty2;
	ret;
      }
    }
    else static assert(0, "todo");
}

void main()
{
    int i = foo();
    printf("i == %d\n", i);
    assert(i == 42);
}

extern(C) int printf(char*, ...);
