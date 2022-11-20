// Tests that out contracts don't interfere with functions returning via sret,
// such as calling the postblit ctor when returning the special __result
// variable.

__gshared int numDtor = 0;

// disabled postblit ctor => only returnable via sret
struct Bar
{
    int v;
    this(this) @disable;
    ~this() { ++numDtor; }
}

Bar rvalue()
out { assert(__result.v == 1); }
do
{
    return Bar(1);
}

Bar nrvo()
out { assert(__result.v == 2); }
do
{
    Bar b = Bar(2);
    return b;
}

void main()
{
    {
        auto a = rvalue();
        assert(a.v == 1);
        assert(numDtor == 0);
    }
    assert(numDtor == 1);

    {
        auto b = nrvo();
        assert(b.v == 2);
        assert(numDtor == 1);
    }
    assert(numDtor == 2);
}
