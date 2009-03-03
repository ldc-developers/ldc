module tangotests.asm4;

extern(C) int printf(char*,...);

void main()
{
    char* stmt = "yay!\n";
    char* fmt = "%s";
    version (D_InlineAsm_X86)
    {
	asm
    	{
		jmp L2;
   	L1:;
		jmp L3;
    	L2:;
		jmp L1;
    	L3:;
		push stmt;
        	call printf;
        	pop EAX;
    	}
    }
    else version(D_InlineAsm_X86_64)
    {
	asm
	{
		jmp L2;
   	L1:;
		jmp L3;
    	L2:;
		jmp L1;
    	L3:;	
		movq	RDI, fmt;
		movq	RSI, stmt;
		xor	AL, AL;
		call	printf;
	}
    }
    printf(fmt,stmt);
}
