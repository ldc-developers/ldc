module slices;

void main()
{
    //char[] a = "hello world";
    //char[5] b = a[0..5];

    //char* cp = a.ptr;
    //char[] c = cp[0..1];
}

char[] first5(char[] str)
{
    char* p = str.ptr;
    return p[0..5];
}

int[] one()
{
    static int i;
    return (&i)[0..1];
}

void[] init()
{
static char c;
return (&c)[0 .. 1];
}

void[] init2()
    {   static char c;

    return (cast(char *)&c)[0 .. 1];
    }