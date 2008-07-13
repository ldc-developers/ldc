module tangotests.asm3;

extern(C) int printf(char*, ...);

void main()
{
    char* fmt = "Hello D World\n";
    printf(fmt);
    asm
    {
        push fmt;
        call printf;
        pop EAX;
    }
}
