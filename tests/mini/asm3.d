module tangotests.asm3;

extern(C) int printf(char*, ...);

void main()
{
    char* fmt = "Hello D World\n";
    printf(fmt);
    version (LLVM_InlineAsm_X86)
    {
	asm
    	{
		push fmt;
        	call printf;
        	pop EAX;
    	}
    }
    else version(LLVM_InlineAsm_X86_64)
    {
        asm
        {
                movq    RDI, fmt;
                xor     AL, AL;
                call    printf;
        }
    }

}
