alias TT(Args...) = Args;

static if (is(__vector(void[16])))
    enum N = 16;
else static if (is(__vector(void[32])))
    enum N = 32;
else
    enum N = 0;

static if (N > 0) void test()
{
    foreach (T; TT!(byte, short, int, long, ubyte, ushort, uint, ulong))
    {
        __vector(T[N / T.sizeof]) a, b, c;
        a *= b;
        a = b * c;
    }
}
