module j;

interface Inter1
{
    int func1();
}

interface Inter2
{
    int func2();
}

class C12 : Inter1, Inter2
{
    int func1()
    {
        return 1;
    }
    int func2()
    {
        return 2;
    }
}

void func(Object c)
{
    auto i1 = cast(Inter1)c;
    assert(i1.func1() == 1);
    auto i2 = cast(Inter2)c;
    assert(i2.func2() == 2);
    auto j1 = cast(Inter1)i2;
    assert(j1.func1() == 1);
    auto j2 = cast(Inter2)i1;
    assert(j2.func2() == 2);
}

void main()
{
    scope c = new C12;
    func(c);
    printf("OK\n");
}

extern(C) int printf(char*,...);
