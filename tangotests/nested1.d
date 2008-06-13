module tangotests.nested1;

void main()
{
    int i = 42;
    assert(i == 42);
    void func()
    {
        assert(i == 42);
    }
    func();
}
