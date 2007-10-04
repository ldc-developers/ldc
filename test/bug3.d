module bug3;

struct S
{
    int[] arr;
    char[5] ch;
}

class C
{
    int[] arr;
    char[4] crs;
}

void main()
{
    S s;
    s.arr = new int[5];
    s.arr[1] = 32;
    assert(s.arr[0] == 0);
    assert(s.arr[1] == 32);
}
