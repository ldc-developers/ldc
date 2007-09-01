version=AndAnd;
version=OrOr;

version(AndAnd)
void andand1()
{
    int a,b;
    a = 4;
    b = 5;
    assert(a == 4);
    assert(b == 5);
    assert(a+b == 9);
    assert(a == 4 && b == 5);
    assert(a != 3 && b == 5);
    assert(a > 2);
    assert(a < 54);
    assert(a < b);
    assert(a > b-2);
    
    int apb = a+b;
    int amb = a*b;
    assert(apb < amb && apb > a);
}

version(OrOr)
void oror1()
{
    int a,b;
    a = 10;
    b = 1000;
    assert(a);
    assert(b);
    assert(a || b);
    assert(a > b || a < b);
}

void main()
{
    printf("Conditionals test\n");
    version(AndAnd) andand1();
    version(OrOr) oror1();
    printf("  SUCCESS\n");
}
