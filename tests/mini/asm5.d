module tangotests.asm5;

extern(C) int printf(char*, ...);

void main()
{
    int i = func();
    printf("%d\n", i);
    assert(i == 42);
}

int func()
{
    asm
    {
    naked;
    mov EAX, 42;
    ret;
    }
}
