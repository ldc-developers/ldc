module interface4;

extern(C) int printf(char*,...);

interface I
{
    void func();
}

interface I2
{
    void func();
}

class C : I,I2
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
    c.func();
    I i = c;
    i.func();
    I2 i2 = c;
    i2.func();
    printf("final %d\n", c.i);
    assert(c.i == 45);
}
