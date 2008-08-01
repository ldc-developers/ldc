module tangotests.marray3;

void main()
{
    int[][][] ma = new int[][][](2,4,3);
    assert(ma.length == 2);
    assert(ma[0].length == 4);
    assert(ma[0][0].length == 3);
    assert(ma[0][1].length == 3);
    assert(ma[0][2].length == 3);
    assert(ma[0][3].length == 3);
    assert(ma[1].length == 4);
    assert(ma[1][0].length == 3);
    assert(ma[1][1].length == 3);
    assert(ma[1][2].length == 3);
    assert(ma[1][3].length == 3);
    ma[0][3][0] = 32;
    ma[1][2][1] = 123;
    ma[0][0][2] = 55;
    assert(ma[0][3][0] == 32);
    assert(ma[1][2][1] == 123);
    assert(ma[0][0][2] == 55);
}
