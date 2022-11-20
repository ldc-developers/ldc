struct Foo
{
    static union Bar
    {
        bool b;
        ulong l;
    }

    Bar bar;
}

static this()
{
    Foo foo = Foo();
}