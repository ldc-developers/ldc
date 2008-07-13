extern(C) int printf(char*, ...);

void main()
{
    int a,b,c;
    a = int.max-1;
    b = 1;
    asm
    {
        mov EAX, a;
        mov ECX, b;
        add EAX, ECX;
        jo Loverflow;
        mov c, EAX;
    }

    printf("c == %d\n", c);
    assert(c == a+b);
    return;

Loverflow:
    assert(0, "overflow");
}
