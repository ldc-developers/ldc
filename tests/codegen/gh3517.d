// Tests EH with two exception types with the same `TypeInfo_Class.name`
// (due to template args not being fully qualified).

// RUN: %ldc -run %s

struct Foo(T)
{
    // name: gh3517.Foo!(S).Foo.MyException
    static class MyException : Exception
    {
        this() { super(null); }
    }
}

void bar()
{
    struct S {}
    Foo!S f;
    throw new f.MyException();
}

void baz()
{
    struct S {}
    Foo!S f2;
    try { bar(); }
    catch (f2.MyException) {} // 1st static TypeDescriptor
}

void main()
{
    struct S {}
    Foo!S f3;

    try
    {
        bar();
    }
    catch (f3.MyException) // 2nd static TypeDescriptor
    {
        assert(0);
    }
    catch (Exception)
    {
        return;
    }

    assert(0);
}
