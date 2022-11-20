extern(C) int printf(const char*, ...);

struct ArrayContainer
{
    int[size_t] _myArray;
    @property auto myArray()
    {
        return _myArray;
    }

    void add(size_t i)
    {
        _myArray[i] = 0;
    }
}

void main()
{
    ArrayContainer x;
    x.add(10);

    foreach(i; x.myArray.keys)
        printf("%d\n", i);
}