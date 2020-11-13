// Tests EH with two exception types with the same `TypeInfo_Class.name`
// (due to template args not being fully qualified).

// RUN: %ldc -c %s

void foo(T)()
{
    // name: gh3501.foo!(S).foo.MyException
    static class MyException : Exception
    {
        this() { super(null); }
    }

    try { throw new MyException(); }
    catch (MyException) {}
}

void bar1()
{
    struct S {}
    foo!S();
}

void bar2()
{
    struct S {}
    foo!S();
}
