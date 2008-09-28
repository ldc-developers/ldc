module mini.const1;

void* g = cast(void*)&foobar;

int foobar()
{
    return 42;
}

void main()
{
    auto fn = cast(int function())g;
    int i = fn();
    assert(i == 42);
}
