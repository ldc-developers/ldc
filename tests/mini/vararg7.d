module tangotests.vararg4;

extern(C) int printf(char*, ...);

struct S
{
    int i;
}

void func(...)
{
    S* sp = cast(S*)_argptr;
    assert(sp.i == 42);
}

void main()
{
    printf("1st:\n");
    {
    S s = S(42);
    func(s);
    }
    printf("ok\n");

    printf("2nd:\n");
    {
    func(S(42));
    }
    printf("ok\n");
}
