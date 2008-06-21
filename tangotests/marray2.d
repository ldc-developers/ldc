module tangotests.marray2;

void main()
{
    int[][] ma = new int[][](2,4);
    assert(ma.length == 2);
    assert(ma[0].length == 4);
    assert(ma[1].length == 4);
    ma[0][3] = 32;
    ma[1][2] = 123;
    ma[0][0] = 55;
    assert(ma[0][3] == 32);
    assert(ma[1][2] == 123);
    assert(ma[0][0] == 55);
}
