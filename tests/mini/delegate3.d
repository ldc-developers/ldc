module bar;

class S
{
    int i;
    final int foo()
    {
        return i;
    }
}

void main()
{
    auto s = new S;
    s.i = 42;
    auto dg = &s.foo;
    assert(dg() == 42);
}
