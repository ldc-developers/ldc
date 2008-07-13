module enum3;

enum GE : ushort
{
    A,B,C
}

void main()
{
    GE e = GE.B;
    size_t s = GE.sizeof;
    assert(e == 1);
    assert(e.sizeof == s);
    assert(s == 2);
}
