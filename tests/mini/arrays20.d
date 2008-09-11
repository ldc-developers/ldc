module mini.arrays20;

int[] foo()
{
    return [2,4,6];
}

int[] bar()
{
    return [1,3,5];
}

void main()
{
    auto a = foo();
    auto b = bar();
    assert(b[0] == 1);
    assert(a[0] == 2);
    assert(b[1] == 3);
    assert(a[1] == 4);
    assert(b[2] == 5);
    assert(a[2] == 6);
}
