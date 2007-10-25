module multiarr1;

void main()
{
    int[16][16] a;
    a[10][13] = 42;
    assert(a[0][0] == 0);
    assert(a[10][13] == 42);
    {assert(*((cast(int*)a)+10*16+13) == 42);}
}
