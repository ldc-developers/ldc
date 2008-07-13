module tangotests.asm4;

extern(C) int printf(char*,...);

void main()
{
    char* fmt = "yay!\n";
    asm
    {
        jmp L2;
    L1:;
        jmp L3;
    L2:;
        jmp L1;
    L3:;
        push fmt;
        call printf;
        pop EAX;
    }
}
