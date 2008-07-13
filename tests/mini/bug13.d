module bug13;

void func1(ubyte[4]* arr)
{
    ubyte* b = &(*arr)[0];
    func2(&(*arr)[1]);
}

void func2(ubyte* ptr)
{
    assert(*ptr == 2);
}

void main()
{
    ubyte[4] arr = [cast(ubyte)1,2,3,4];
    func1(&arr);
}
