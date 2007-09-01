module condexp;

int f()
{
    return 42;
}

void main()
{
    int i = f() < 25 ? -1 : 1;
    /*int j = f() > 25 ? 1 : -1;
    assert(i);
    assert(!j);*/
}
