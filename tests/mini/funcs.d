extern(C) int printf(char*, ...);

void main()
{
    printf("Testing functions\n");
    int i = 5;
    assert(a(i) == 110);
    assert(i == 11);

    S s;
    s.i = 5;
    d(s);
    assert(s.i == 5);
    e(s);
    assert(s.i == 6);

    printf("  SUCCESS\n");
}

int a(ref int i)
{
    i*=2;
    return b(i);
}

int b(ref int i)
{
    i++;
    return c(i);
}

int c(int i)
{
    return i*10;
}

struct S
{
    int i;
}

void d(S s)
{
    s.i++;
}

void e(ref S s)
{
    s.i++;
}
