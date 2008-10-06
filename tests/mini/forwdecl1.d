struct Foo;

Foo* foo()
{
    return null;
}

void main()
{
    Foo* f = foo();
    assert(f is null);
}
