module asm1;

extern(C) int printf(char*, ...);

void main()
{
    version(D_InlineAsm_X86)
    {
	int x;
	asm
	{
	    mov EAX, 42;
	    mov x, EAX;
	}
	printf("x = %d\n", x);
    }
    else version(D_InlineAsm_X86_64)
    {
        long x;
        asm
        {
            mov RAX, 42L;
            mov x, RAX;
        }
        printf("x = %ld\n", x);
    }
    else
    {
        static assert(0, "no inline asm for this platform yet");
    }
}
