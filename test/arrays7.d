module arrays7;

extern(C) int printf(char*, ...);

pragma(LLVM_internal, "notypeinfo")
struct S
{
    int i;
    float f;
    long l;

    void print()
    {
        printf("%d %f %lx\n", i, f, l);
    }
}

void main()
{
    S[] arr;
    S s;
    assert(arr.length == 0);
    arr ~= s;
    assert(arr.length == 1);
    arr ~= S(1,2.64,0xFFFF_FFFF_FFFF);
    assert(arr.length == 2);
    arr[0].print();
    arr[1].print();
    assert(arr[1].i == 1);
    assert(arr[1].f > 2.63 && arr[1].f < 2.65);
    assert(arr[1].l == 0xFFFF_FFFF_FFFF);
}
