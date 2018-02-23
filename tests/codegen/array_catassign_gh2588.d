// RUN: %ldc -run %s

int work(ref int[] array)
{
    array ~= 123;
    return 456;
}

void main()
{
    int[] array;
    array ~= work(array);
    assert(array == [ 123, 456 ]);
}
