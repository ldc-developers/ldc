extern(C) int printf(char*, ...);

version (D_InlineAsm_X86)
    version = InlineAsm_X86_Any;
version (D_InlineAsm_X86_64)
    version = InlineAsm_X86_Any;

void main()
{
    int a,b,c;
    a = int.max-1;
    b = 5;
    version (InlineAsm_X86_Any)
    {
        asm
        {
            mov EAX, a;
            mov ECX, b;
            add EAX, ECX;
            jo Loverflow;
            mov c, EAX;
        }
    }
    printf("a == %d\n", a);
    printf("b == %d\n", b);
    printf("c == %d\n", c);
    assert(c == c);
    return;

Loverflow:
int y=0;
    //assert(0, "overflow");
}
