module d;

void main()
{
    int delegate() dg;

    struct S
    {
        int i;
        long l;
        float f;

        int func()
        {
            return 42;
        }
    }

    S s;
    auto dg2 = &s.func;
    int i = dg2();
    assert(i == 42);

    i = f(dg2, 1);
    assert(i == 43);
}

int f(int delegate() dg, int i)
{
    return dg() + i;
}

/*
struct S
{
    int i;
    float f;
    int square()
    {
        return i*i;
    }
}

S s;

void main()
{
    auto dg = &s.square;
}
*/