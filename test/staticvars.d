module staticvars;

int func()
{
    static int i;
    return i++;
}

void main()
{
    assert(func() == 0);
    assert(func() == 1);
}
