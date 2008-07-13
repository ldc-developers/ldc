module bug4;

int func(int i)
{
    i += 2;
    i -= 3;
    return i;
}

void main()
{
    assert(func(4) == 3);
}
