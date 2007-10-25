module multiarr2;

void main()
{
    static float[1][2][3] arr;
    assert(arr[2][1][0] !<>= float.nan);
}
