extern(C) int printf(char*, ...);
void main()
{
    void* i;
    asm
    {
        mov EAX, FS:4;
        mov i, EAX;
    }
    printf("FS:4 = %p\n", i);
}
