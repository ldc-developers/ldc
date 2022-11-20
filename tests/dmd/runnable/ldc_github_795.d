debug import core.stdc.stdio;

struct Int
{
    static int count;

    this(int)
    {
        ++count;
        debug printf("CONSTRUCT count = %d\n", count);
    }

    this(this)
    {
        ++count;
        debug printf("COPY count = %d\n", count);
    }

    ~this()
    {
        --count;
        debug printf("DESTROY count = %d\n", count);
    }
}

struct Only
{
    Int front;
    bool empty;

    void popFront()
    {
        empty = true;
    }
}

struct Map {
    Only r;
    bool empty;

    auto front() @property
    {
        return Only(r.front);
    }

    void popFront()
    {
        empty = true;
    }
}

void test1()
{
    {
        auto sm = Map(Only(Int(42)));
        bool condition = !sm.empty && sm.front.empty;
    }
    assert(Int.count == 0);
}

void test2()
{
    {
        auto sm = Map(Only(Int(42)));
        bool condition = sm.empty || sm.front.empty;
    }
    assert(Int.count == 0);
}

void test3()
{
    {
        auto sm = Map(Only(Int(42)));
        bool condition = sm.empty ? false : sm.front.empty;
    }
    assert(Int.count == 0);
}

void test4()
{
    {
        auto sm = Map(Only(Int(42)));
        bool condition = !sm.empty ? sm.front.empty : false;
    }
    assert(Int.count == 0);
}

void main()
{
    test1();
    test2();
    test3();
    test4();
}
