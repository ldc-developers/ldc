int dtor;

struct DestroyMe
{
    ~this() { ++dtor; }

    int opApply(in int delegate(int item) dg)
    {
        throw new Exception("Here we go!");
    }
}

void test1()
{
    dtor = 0;
    try {
        foreach (item; DestroyMe()) {}
    } catch (Exception) {}
    assert(dtor == 1);

    dtor = 0;
    try {
        auto lvalue = DestroyMe();
        foreach (item; lvalue) {}
    } catch (Exception) {}
    assert(dtor == 1);
}

/******************************************/

struct DoNotDestroyMe
{
    ~this()
    {
        assert(0);
    }
}

DoNotDestroyMe doNotDestroyMe()
{
    throw new Exception("Here we go!");
    return DoNotDestroyMe();
}

void fun(E)(lazy E exp)
{
    try {
      exp();
    } catch (Exception) {}
}

void test2()
{
    fun(doNotDestroyMe());
}

void main()
{
    test1();
    test2();
}
