module interface3;

interface I
{
    void func();
}

class C : I
{
    int i = 42;
    override void func()
    {
        printf("hello %d\n", i);
        i++;
    }
}

void main()
{
    scope c = new C;
    {c.func();}
    {
        I i = c;
        {i.func();}
    }
    {printf("final %d\n", c.i);}
    {assert(c.i == 44);}
}
