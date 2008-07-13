module c;

void main()
{
    ushort a = 0xFFF0;
    ushort b = 0x0FFF;
    auto t = a & b;
    a &= b;
    assert(t == a);
}
