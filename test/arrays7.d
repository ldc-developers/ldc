module arrays7;

struct S
{
    int i;
    float f;
    long l;
}

void main()
{
    S[] arr;
    S s;
    arr ~= s;
    arr ~= S(1,2.64,0xFFFF_FFFF_FFFF);
    assert(arr[1].i == 1);
    assert(arr[1].f > 2.63 && arr[1].f < 2.65);
    assert(arr[1].l == 0xFFFF_FFFF_FFFF);
}
