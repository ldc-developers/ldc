int numPostblit = 0, numDtor = 0;

struct S
{
    int v;
    this(this) { ++numPostblit; }
    ~this() { ++numDtor; }
}

void foo()
{
    S[4] sa = [ S(1), S(2), S(3), S(4) ];

    // helper to generate a slice rvalue
    static S[] toSlice(ref S[4] sa) { return sa[1..$]; }

    S[3] r = toSlice(sa);
}

void main()
{
    foo();
    assert(numPostblit == 3);
    assert(numDtor == 7);
}
