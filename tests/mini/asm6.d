extern(C) int printf(char*, ...);

void main()
{
    int a,b,c;
    a = int.max-1;
    b = 5;
    version (D_InlineAsm_X86)
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
    else version (D_InlineAsm_X86_64)
    {
	asm
	{
		movq RDX, a;
        	movq RAX, b;
        	add RDX, RAX;
        	jo Loverflow;
        	movq c, RDX;
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
