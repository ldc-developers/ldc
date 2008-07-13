module tangotests.mem3;

void main()
{
    int[] arr;
    arr ~= [1,2,3];
    assert(arr[0] == 1);
    assert(arr[1] == 2);
    assert(arr[2] == 3);
    delete arr;
    assert(arr is null);
}
