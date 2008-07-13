module tangotests.asm2;

extern(C) int printf(char*, ...);

int main()
{
    int i = 40;
    int j = 2;
    asm
    {
        mov EAX, i;
        mov EBX, j;
        add EAX, EBX;
        mov i, EAX;
    }
    printf("42 = %d\n", i);
    assert(i == 42);
    return 0;
}
