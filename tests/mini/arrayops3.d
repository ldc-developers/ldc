void main()
{
    int[4] a = [1,2,3,4];
    int[4] b = [5,6,7,8];
    a[] += b[] / 2;
    assert(a[0] == 3);
    assert(a[1] == 5);
    assert(a[2] == 6);
    assert(a[3] == 8);
}
