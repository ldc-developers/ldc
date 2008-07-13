module tangotest.marray1;

void main()
{
    int[][] arr;
    int[] a = [1,2];
    int[] b = [6,7,8,9];
    arr ~= a;
    arr ~= b;
    assert(a.length == 2);
    assert(b.length == 4);
    assert(arr.length == 2);
    assert(arr[0][0] == 1);
    assert(arr[0][1] == 2);
    assert(arr[1][0] == 6);
    assert(arr[1][1] == 7);
    assert(arr[1][2] == 8);
    assert(arr[1][3] == 9);
}
