module d;
/*
void main()
{
    int delegate() dg;
    int i = dg();

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
    i = dg2();

    i = f(dg2, 1);
}

int f(int delegate() dg, int i)
{
    return dg() + i;
}
*/

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
