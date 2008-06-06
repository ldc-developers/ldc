module tangotests.asm2;

extern(C) int printf(char*, ...);

int main()
{
    int i = 40;
    asm
    {
        mov EAX, i;
        add EAX, 2;
        mov i, EAX;
    }
    printf("42 = %d\n", i);
    assert(i == 42);
    return 0;
}
