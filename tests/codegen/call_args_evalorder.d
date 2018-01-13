// RUN: %ldc -run %s

void checkInt(int a, int b, int c)
{
    assert(a == 1);
    assert(b == 2);
    assert(c == 3);
}

int factory(ref int a)
{
    a += 2;
    return 2;
}

void main()
{
    int a = 1;
    checkInt(a, factory(a), a);
}
