module calls1;
import std.stdarg;
void main()
{
    {int a = byVal1(3);}
    {int a = void; byRef1(a);}
    {char[] c = void; refType(c);}
    {char[] c = void; refTypeByRef(c);}
    {S s = void; structByVal(s);}
    {S s = void; structByRef(s);}
    {S s = void; structByPtr(&s);}
    {printf("c-varargs %d %d %d\n", 1,2,3);}
    {int i=3; float f=24.7; dvararg(i,f);}
    {char[] s = "hello"; dvarargRefTy(s);}
    {char[] ss = "hello world!"; dvarargRefTy(ss);}
}

int byVal1(int a)
{
    return a;
}

void byRef1(ref int a)
{
    a = 3;
}

void refType(char[] s)
{
}

void refTypeByRef(ref char[] s)
{
}

struct S
{
    float f;
    double d;
    long l;
    byte b;
}

void structByVal(S s)
{
}

void structByRef(ref S s)
{
}

void structByPtr(S* s)
{
}

void dvararg(...)
{
    printf("%d %.1f\n", va_arg!(int)(_argptr), va_arg!(float)(_argptr));
}

void dvarargRefTy(...)
{
    char[] s = va_arg!(char[])(_argptr);
    printf("%.*s\n", s.length, s.ptr);
}
