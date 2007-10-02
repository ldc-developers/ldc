module classes7;

class C
{
    int i=0;
    void f()
    {
        i=42;
    }
    void g()
    {
        f();
    }
}

void main()
{
    scope c = new C;
    c.g();
    assert(c.i == 43);
}
