module nested3;

void main()
{
    int i;
    void test()
    {
        i = 3;
    }
    test();
    assert(i == 3);
}
