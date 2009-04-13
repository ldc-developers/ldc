module tangotests.asm1_1;

extern(C) int printf(char*, ...);

int main()
{
    int i = 12;
    int* ip = &i;
    printf("%d\n", i);
    version (D_InlineAsm_X86)
    {
        asm
        {
            mov ECX, ip;
            mov EAX, [ECX];
            add EAX, 8;
            mul EAX, EAX;
            mov [ECX], EAX;
        }
    }
    else version (D_InlineAsm_X86_64)
    {
        asm
        { 
            movq RCX, ip;
            mov EAX, [RCX];
            add EAX, 8;
            imul EAX, EAX;
            mov [RCX], EAX;
        }
    }
    printf("%d\n", i);
    assert(i == 400);
    return 0;
}
