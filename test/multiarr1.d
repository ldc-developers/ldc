module multiarr1;

void main()
{
    int[16][16] a;
    a[10][13] = 42;
    //assert(a[0][0] == 0);
    //assert(a[10][13] == 42);
    {
        int* l = cast(int*)a;
        l += 10*16+13;
        assert(*l == 42);
    }
}
