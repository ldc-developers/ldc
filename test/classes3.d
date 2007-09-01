class C
{
    int c;
    long f(long l)
    {
        return l;
    }
}

class D : C
{
    int d;
    override long f(long l)
    {
        return l*2;
    }
}

void main()
{
    scope c = new C;
    assert(c.f(25L) == 25);
    scope d = new D;
    assert(d.f(25L) == 50);
    C cd = d;
    assert(cd.f(25L) == 50);
    assert(func(d,25L) == 50);
}

long func(C c, long l)
{
    return c.f(l);
}
