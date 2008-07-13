module union5;

pragma(LLVM_internal, "notypeinfo")
{
    union S
    {
        T t;
        U u;
        uint i;
        struct {
            ushort sl,sh;
        }
    }

    struct T
    {
        int i;
    }

    struct U
    {
        float f;
    }
}

void main()
{
    S s;
    assert(s.t.i == 0);
    assert(s.u.f == 0);
    s.t.i = -1;
    assert(s.i == 0xFFFF_FFFF);
    float f = 3.1415;
    s.u.f = f;
    uint pi = *cast(uint*)&f;
    assert(s.i == pi);
    assert(s.sl == (pi&0xFFFF));
    assert(s.sh == (pi>>>16));
}
