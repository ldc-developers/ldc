module multiarr3;

void main()
{
    static int[2][2] arr = [[1,2],[3,4]];
    assert(arr[0][0] == 1);
    assert(arr[0][1] == 2);
    assert(arr[1][0] == 3);
    assert(arr[1][1] == 4);
}
