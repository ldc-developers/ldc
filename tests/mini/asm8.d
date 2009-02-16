const float  one_f = 1;
const double one_d = 1;
const real   one_r = 1;

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
        asm { movss XMM0, [one_f]; }
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
        asm { movsd XMM0, [one_d]; }
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

ifloat oneif()
{
    version(X86)
    {
        asm { fld1; }
    }
    else version (X86_64)
    {
        asm { movss XMM0, [one_f]; }
    }
    else static assert(0, "todo");
}

idouble oneid()
{
    version(X86)
    {
        asm { fld1; }
    }
    else version (X86_64)
    {
        asm { movsd XMM0, [one_d]; }
    }
    else static assert(0, "todo");
}

ireal oneir()
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


const float two_f = 2;

cfloat cf()
{
    version(X86)
    {
        asm { fld1; fld two_f; }
    }
    else version (X86_64)
    {
        version(all) {
            asm
            {
                movss XMM0, [one_f];
                movss XMM1, [two_f];
            }
        } else {
            // Code for when LDC becomes ABI-compatible with GCC
            // regarding cfloat returns.
            asm {
                movd EAX, [one_f];
                movd ECX, [two_f];
                
                // invalid operand size :(
                //shl RCX, 32;
                //or RAX, RCX;
                
                pushq RAX;
                mov [RSP + 4], EAX;
                popq RAX;
                
                movd XMM0, RAX;
            }
        }
    }
    else static assert(0, "todo");
}

cfloat cf2()
{
    version(X86)
    {
        asm
        {
            naked;
            fld1;
            fld two_f;
            ret;
        }
    }
    else version (X86_64)
    {
        version(all) {
            asm
            {
                naked;
                movss XMM0, [one_f];
                movss XMM1, [two_f];
                ret;
            }
        } else {
            // Code for when LDC becomes ABI-compatible with GCC
            // regarding cfloat returns.
            asm {
                naked;
                mov EAX, [one_f];
                mov ECX, [two_f];
                
                // invalid operand size :(
                //shl RCX, 32;
                //or RAX, RCX;
                
                pushq RAX;
                mov [RSP + 4], EAX;
                popq RAX;
                
                movd RAX, XMM0;
                ret;
            }
        }
    }
    else static assert(0, "todo");
}


const double two_d = 2;

cdouble cd()
{
    version(X86)
    {
        asm { fld1; fld two_d; }
    }
    else version (X86_64)
    {
        asm
        {
            leaq RAX, [one_d];
            leaq RCX, [two_d];
            movsd XMM0, [RAX];
            movsd XMM1, [RCX];
        }
    }
    else static assert(0, "todo");
}

cdouble cd2()
{
    version(X86)
    {
        asm
        {
            naked;
            fld1;
            fld two_d;
            ret;
        }
    }
    else version (X86_64)
    {
        asm
        {
            naked;
            movsd XMM0, [one_d];
            movsd XMM1, [two_d];
        }
    }
    else static assert(0, "todo");
}


const real two_r = 2.0;

creal cr()
{
    version(X86)
    {
        asm { fld1; fld two_r; }
    }
    else version (X86_64)
    {
        asm { fld two_r; fld1; }
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
            fld two_r;
            ret;
        }
    }
    else version (X86_64)
    {
        asm
        {
            naked;
            fld two_r;
            fld1;
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
    
    assert(oneif() == 1i);
    assert(oneid() == 1i);
    assert(oneir() == 1i);
    
    assert(cf() == 1+2i);
    assert(cf2() == 1+2i);
    
    assert(cd() == 1+2i);
    assert(cd2() == 1+2i);
    
    assert(cr() == 1+2i);
    assert(cr2() == 1+2i);
    
    assert(vp() == cast(void*)0x80);
    assert(aa() is gaa);
    assert(ob() is gobj);
    assert(str() == "hello world");
    assert(dg()() == "object.Object");
}

extern(C) int printf(char*, ...);
