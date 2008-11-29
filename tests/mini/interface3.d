module interface3;

extern(C) int printf(char*,...);

interface I
{
    void func();
}

class C : I
{
    int i = 42;
    override void func()
    {
        printf("hello %d from %p\n", i, this);
        i++;
    }
}

void main()
{
    auto c = new C;
    {c.func();}
    {
        I i = c;
        {i.func();}
    }
    {printf("final %d\n", c.i);}
    {assert(c.i == 44);}
}
