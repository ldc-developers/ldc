int foo()
{
    version(X86)
    {
        asm { mov EAX, 42; }
    }
    else version (X86_64)
    {
        asm { mov EAX, 42; }
    }
    else static assert(0, "todo");
}

ulong bar()
{
    version(X86)
    {
        asm { mov EDX, 0xAA; mov EAX, 0xFF; }
    }
    else version (X86_64)
    {
        asm { movq RAX, 0xFF; }
    }
    else static assert(0, "todo");
}

float onef()
{
    version(X86)
    {
        asm { fld1; }
    }
    else version (X86_64)
    {
        asm { fld1; }
    }
    else static assert(0, "todo");
}

double oned()
{
    version(X86)
    {
        asm { fld1; }
    }
    else version (X86_64)
    {
        asm { fld1; }
    }
    else static assert(0, "todo");
}

real oner()
{
    version(X86)
    {
        asm { fld1; }
    }
    else version (X86_64)
    {
        asm { fld1; }
    }
    else static assert(0, "todo");
}


real two = 2.0;

creal cr()
{
    version(X86)
    {
        asm { fld1; fld two; }
    }
    else version (X86_64)
    {
        asm { fld1; fld two; }
    }
    else static assert(0, "todo");
}

creal cr2()
{
    version(X86)
    {
        asm
        {
            naked;
            fld1;
            fld two;
            ret;
        }
    }
    else version (X86_64)
    {
    asm
        {
            naked;
            fld1;
            fld two;
            ret;
        }
    }
    else static assert(0, "todo");
}

void* vp()
{
    version(X86)
    {
        asm { mov EAX, 0x80; }
    }
    else version (X86_64)
    {
        asm { movq RAX, 0x80; }
    }
    else static assert(0, "todo");
}

int[int] gaa;

int[int] aa()
{
    version(X86)
    {
        asm { mov EAX, gaa; }
    }
    else version (X86_64)
    {
        asm { movq RAX, gaa; }
    }
    else static assert(0, "todo");
}

Object gobj;

Object ob()
{
    version(X86)
    {
        asm { mov EAX, gobj; }
    }
    else version (X86_64)
    {
        asm { movq RAX, gobj; }
    }
    else static assert(0, "todo");
}

char[] ghello = "hello world";

char[] str()
{
    version(X86)
    {
        asm { lea ECX, ghello; mov EAX, [ECX]; mov EDX, [ECX+4]; }
    }
    else version (X86_64)
    {
        asm { movq RAX, [ghello]; movq RDX, [ghello]+8; }
    }
    else static assert(0, "todo");
}

char[] delegate() dg()
{
    version(X86)
    {
        asm { mov EAX, gobj; lea EDX, Object.toString; }
    }
    else version (X86_64)
    {
        asm { movq RAX, [gobj]; leaq RDX, Object.toString; }
    }
    else static assert(0, "todo");
}

void main()
{
    gaa[4] = 5;
    gobj = new Object;
    auto adg = &gobj.toString;

    assert(foo() == 42);
    version(X86)
    {
        assert(bar() == 0x000000AA000000FF);
    } 
    else version (X86_64)
    {
        assert(bar() == 0x00000000000000FF);
    }
    assert(onef() == 1);
    assert(oned() == 1);
    assert(oner() == 1);
    assert(cr() == 1+2i);
    assert(cr2() == 1+2i);
    assert(vp() == cast(void*)0x80);
    assert(aa() is gaa);
    assert(ob() is gobj);
    assert(str() == "hello world");
    assert(dg()() == "object.Object");
}

extern(C) int printf(char*, ...);
