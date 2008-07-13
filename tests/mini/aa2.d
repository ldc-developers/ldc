module aa2;

void main()
{
    long[float] aa;
    long* p = 2.0f in aa;
    assert(!p);
    aa[4f] = 23;
    p = 4f in aa;
    assert(p);
    assert(*p == 23);
}
