module tangotests.asm7;

// test massive label collisions (runtime uses Loverflow too)
extern(C) int printf(char*, ...);

void main()
{
    int a = add(1,2);
    int s = sub(1,2);
    assert(a == 3);
    assert(s == -1);
}

int add(int a, int b)
{
    int res;
    version (D_InlineAsm_X86)
    {
	asm
    	{
		mov EAX, a;
        	add EAX, b;
        	jo Loverflow;
        	mov res, EAX;
    	}
    }
    else version (D_InlineAsm_X86_64)
    {
	asm
	{
		mov EAX, a;
        	add EAX, b;
        	jo Loverflow;
        	mov res, EAX;
	}		
    }
    printf("%d\n",res);
    return res;
Loverflow:
    assert(0, "add overflow");
}

int sub(int a, int b)
{
    int res;
    version (D_InlineAsm_X86)
    {
    	asm
    	{
		mov EAX, a;
        	sub EAX, b;
        	jo Loverflow;
        	mov res, EAX;
    	}
    }
    else version (D_InlineAsm_X86_64)
    {
	asm
	{
		mov EAX, a;
        	sub EAX, b;
        	jo Loverflow;
        	mov res, EAX;
	}		
    }
    printf("%d\n",res);
    return res;
Loverflow:
    assert(0, "sub overflow");
    int x;
}
