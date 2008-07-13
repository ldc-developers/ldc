module multiarr1;

void main()
{
    int[16][16] a;
    assert(a[0][0] == 0);
    assert(a[0][1] == 0);
    assert(a[0][2] == 0);
    assert(a[0][3] == 0);
    assert(a[10][13] == 0);
    assert(a[15][15] == 0);
    a[10][13] = 42;
    assert(a[0][0] == 0);
    assert(a[10][13] == 42);
    assert(a[15][15] == 0);
    {
        int* l = cast(int*)a;
        l += 10*16+13;
        assert(*l == 42);
    }
}
