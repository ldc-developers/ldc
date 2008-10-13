void main()
{
    int[4] a = [1,2,3,4];
    int[4] b = [5,6,7,8];
    a[] += b[];
    assert(a[0] == 6);
    assert(a[1] == 8);
    assert(a[2] == 10);
    assert(a[3] == 12);
}
