module bug27;

int func(int a, int b)
{
    if (a == b)
        return 0;
    else if (a < b)
        return -1;
    else
        return 1;
}

void main()
{
    int i = func(3,4);
    assert(i == -1);
}
