int foo(int op)(int a, int b)
{
    version(X86)
    {
    const OP = (op == '+') ? "add" : "sub";
    version (Windows)
    {
    	asm { naked; }
    	mixin("asm{push EBP;mov EBP,ESP;sub ESP,8;mov ECX,[EBP+8];"~OP~" EAX, ECX;add ESP,8;pop EBP;}");
    	asm { ret; }
	} else
	{
    	asm { naked; }
    	mixin("asm{"~OP~" EAX, [ESP+4];}");
    	asm { ret 4; }
	}
    }
    else version(X86_64)
    {
    const OP = (op == '+') ? "add" : "sub";
    asm { naked; }
    mixin("asm{"~OP~" ESI,EDI; mov EAX, ESI;}");
    asm { ret; }
    }
    else static assert(0, "todo");
}

void main()
{
        int i = foo!('+')(2, 4);
        assert(i == 6);
        i = foo!('-')(2, 4);
        assert(i == 2);
}
