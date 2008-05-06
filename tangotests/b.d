class C
{
    int a;
    union
    {
        int i;
        double d;
    }
    int z;
}

void func()
{
    scope c = new C;
    access1(c);
    assert(c.i == 42);
    access2(c);
    assert(c.d == 2.5);
}

void access1(C c)
{
    c.i = 42;
}

void access2(C c)
{
    c.d = 2.5;
}

void main()
{
    func();
}
