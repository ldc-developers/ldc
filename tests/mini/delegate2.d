module mini.delegate2;

void main()
{
    int foo = 42;
    int bar()
    {
        return foo;
    }
    int delegate() dg = &bar;
    assert(dg() == foo);
}
