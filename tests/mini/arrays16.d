module mini.arrays16;

void main()
{
    intarrays!(byte)();
    intarrays!(ubyte)();
    intarrays!(short)();
    intarrays!(ushort)();
    intarrays!(int)();
    intarrays!(uint)();
    intarrays!(long)();
    intarrays!(ulong)();
}

void intarrays(T)()
{
    T[] ia = [cast(T)1,2,3,4];
    T[] ib = [cast(T)1,2,3,4];
    T[] ic = [cast(T)1,2,3];
    T[] id = [cast(T)1,2,3,4,5];

    assert(ia == ia);
    assert(ia == ib);
    assert(ia != ic);
    assert(ia != id);
    assert(ia > ic);
    assert(ia !< ic);
    assert(ia < id);
    assert(ia !> id);
}
