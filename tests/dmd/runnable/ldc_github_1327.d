void main()
{
    int v = 3;
    static int inc(ref int v) { ++v; return 10; }

    int r = v + inc(v);
    assert(r == 3 + 10);
    assert(v == 4);

    v *= inc(v) + v;
    assert(v == (4+1) * (10 + 5));
}
