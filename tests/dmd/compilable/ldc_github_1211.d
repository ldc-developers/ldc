struct Foo
{
    long baz;
}

Foo foo(long x)
{
    struct Bar
    {
        long y;
    }

    return cast(Foo)Bar(x);
}
