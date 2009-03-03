module mini.callingconv1;

extern(C) int printf(char*, ...);

float foo(float a, float b)
{
    return a + b;
}

void main()
{
    float a = 1.5;
    float b = 2.5;
    float c;

    version(D_InlineAsm_X86)
    {
	asm
    	{
		mov EAX, [a];
        	push EAX;
        	mov EAX, [b];
        	push EAX;
        	call foo;
        	fstp c;
    	}
    }
    else version(D_InlineAsm_X86_64)
    {
    	asm
    	{
		movss XMM0, [a];
		movss XMM1, [b];
        	call foo;
		movss [c], XMM0;
    	}
    }
    printf("%f\n", c);

    assert(c == 4.0);
    
    printf("passed %f\n", c);
}
