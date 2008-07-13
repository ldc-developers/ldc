
extern(C) int printf(char*, ...);

struct S
{
    int i;
    int square()
    {
        return i*i;
    }
    int plus(int a)
    {
        return i + a;
    }
    int minus(int a)
    {
        return i - a;
    }
    int delegate(int) get(char op)
    {
        int delegate(int) rval;
        if (op == '+')
            rval = &plus;
        else if (op == '-')
            rval = &minus;
        return rval;
    }
}

int calldg1(int delegate(int) dg, int i)
{
    return dg(i);
}

void delegate() retdg()
{
    void delegate() dg;
    return dg;
}

void getretdg()
{
    void delegate() dg;
    dg = retdg();
}

class C
{
    int i;
    void m()
    {
        i = 42;
    }
}

void getclassdg()
{
    scope c = new C;
    void delegate() dg = &c.m;
    assert(c.i != 42);
    dg();
    assert(c.i == 42);
}

void main()
{
    printf("Delegate test\n");
    S s = S(4);
    
    auto dg = &s.square;
    //assert(dg() == 16);
    //dg();
    
    /*auto dg1 = &s.plus;
    assert(dg1(6) == 10);
    
    auto dg2 = &s.minus;
    assert(calldg1(dg2,30) == -26);
    
    auto dg3 = s.get('+');
    assert(dg3(16) == 20);
    
    getretdg();
    getclassdg();*/
    
    printf("  SUCCESS\n");
}
