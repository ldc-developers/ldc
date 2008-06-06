module tangotests.asm1;

extern(C) int printf(char*, ...);

int main()
{
    int i = 12;
    int* ip = &i;
    printf("%d\n", i);
    asm
    {
        mov EBX, ip;
        mov EAX, [EBX];
        add EAX, 8;
        mul EAX, EAX;
        mov [EBX], EAX;
    }
    printf("%d\n", i);
    assert(i == 400);
    return 0;
}
