module globals2;

template Bool(bool b)
{
    const bool Bool = b;
}

void main()
{
    assert(Bool!(true));
    assert(!Bool!(false));
}
