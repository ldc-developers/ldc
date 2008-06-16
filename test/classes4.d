extern(C) int printf(char*, ...);

class A
{
    int i = 42;
    double df = 3.1415;
    this()
    {
    }
    char[] toString()
    {
        return "A:Object";
    }
}

class B : A
{
    ubyte b;
    char[] toString()
    {
        return "B:A";
    }
}

void main()
{
    scope a = new A;
    char[] as = a.toString;
    {printf("a.toString = '%.*s'\n", as.length, as.ptr);}

    Object o = a;
    char[] os = o.toString;
    {printf("o.toString = '%.*s'\n", os.length, os.ptr);}

    scope b = new B;
    char[] bs = b.toString;
    {printf("b.toString = '%.*s'\n", bs.length, bs.ptr);}
}
