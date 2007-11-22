module classinfo3;

class C
{
    int i;
    float f;
    long l;
    int j;
}

void main()
{
    auto c = C.classinfo;
    assert(c.offTi !is null);
    assert(c.offTi.length == 4);

    size_t base = 2*size_t.sizeof;

    assert(c.offTi[0].offset == base);
    assert(c.offTi[0].ti == typeid(int));
    assert(c.offTi[1].offset == base+4);
    assert(c.offTi[1].ti == typeid(float));
    assert(c.offTi[2].offset == base+8);
    assert(c.offTi[2].ti == typeid(long));
    assert(c.offTi[3].offset == base+16);
    assert(c.offTi[3].ti == typeid(int));
}
