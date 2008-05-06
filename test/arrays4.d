module arrays4;

void main()
{
    int[] arr;
    arr ~= 3;
    assert(arr.length == 1);
    assert(arr[0] == 3);
    arr ~= 5;
    assert(arr.length == 2);
    assert(arr[0] == 3);
    assert(arr[1] == 5);
}
