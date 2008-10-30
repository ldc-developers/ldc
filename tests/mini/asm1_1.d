module tangotests.asm1_1;

extern(C) int printf(char*, ...);

int main()
{
    int i = 12;
    int* ip = &i;
    printf("%d\n", i);
    version (LLVM_InlineAsm_X86)
    {
	asm
    	{
		mov EBX, ip;
        	mov EAX, [EBX];
        	add EAX, 8;
        	mul EAX, EAX;
        	mov [EBX], EAX;
    	}
    }
    else version (LLVM_InlineAsm_X86_64)
    {
	asm
	{ 
		movq RCX, ip;
		movq RAX, [RCX];
		add RAX, 8;
		imul RAX, RAX;
		movq [RCX], RAX;
	}
    }
    printf("%d\n", i);
    assert(i == 400);
    return 0;
}
