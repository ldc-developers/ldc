module nested2;

void func(ref int i)
{
    delegate {
        assert(i == 3);
        i++;
    }();
}

void main()
{
    int i = 3;
    func(i);
    assert(i == 4);
}
