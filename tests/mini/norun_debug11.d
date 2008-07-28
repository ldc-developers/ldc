module mini.norun_debug11;

class C
{
}

class D : C
{
    int i = 42;
}

class E : D
{
    float fp = 3.14f;
}

class F : E
{
    F f;
}

void main()
{
    auto c = new C;
    auto d = new D;
    auto e = new E;
    auto f = new F;

    auto ci = c.classinfo;

    int* fail;
    *fail = 0;
}
