// RUN: %ldc -run %s

void main()
{
    int[2][1] arr;
    assert(&(arr[0][0]) !is &(arr[0][1]));
    ubyte RR = 0;
    assert(&(arr[RR][0]) !is &(arr[RR][1]));
}
