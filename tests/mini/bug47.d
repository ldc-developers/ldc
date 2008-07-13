module bug47;

bool func(bool a, bool b)
{
    if (a) b = false;
    return b;
}

void main()
{
    assert(func(0,0) == 0);
    assert(func(0,1) == 1);
    assert(func(1,0) == 0);
    assert(func(1,1) == 0);
}
