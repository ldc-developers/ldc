// Tests EH with exception types with previously identical `TypeInfo_Class.name`
// (due to template args not being fully qualified - fixed since D v2.098).

// RUN: %ldc -run %s

struct Foo(T)
{
    static class MyException : Exception
    {
        this() { super(null); }
    }
}

void throwAndCatch()
{
    struct S {}
    Foo!S f;

    try { throw new f.MyException(); }
    catch (f.MyException) {} // issue 3501: 1st static TypeDescriptor
}

void doThrow()
{
    struct S {}
    Foo!S f1;
    throw new f1.MyException();
}

void throwAndDontCatchAnother()
{
    struct S {}
    Foo!S f2;

    try { doThrow(); }
    catch (f2.MyException) { assert(0); } // issue 3501: 2nd static TypeDescriptor
    catch (Exception) { return; }
    assert(0);
}

void main()
{
    throwAndCatch();
    throwAndDontCatchAnother();
}
