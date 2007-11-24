module interface2;

interface A
{
    void a();
}

interface B
{
    void b();
}

class C : A,B
{
    int i = 0;
    override void a()
    {
        printf("hello from C.a\n");
    }
    override void b()
    {
        printf("hello from C.b\n");
    }
}

void main()
{
    scope c = new C;
    {c.a();
    c.b();}
    {A a = c;
    a.a();}
    {B b = c;
    b.b();}
}
