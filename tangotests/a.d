class Foo
{
    this(int j)
    {
        i = pi = j;
    }

    int i;

private:

    int pi;
}

class Bar : Foo
{
    this(int j)
    {
        super(j);
        baz = 42;
    }

    int baz;
}

void func()
{
    auto bar = new Bar(12);
}
