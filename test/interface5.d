module interface5;

extern(C) int printf(char*,...);

interface I
{
    void func();
}

class C : I
{
    int i;
    void func()
    {
        printf("C\n");
        i++;
    }
}

void main()
{
    C c = new C;
    c.func();
    {
        I i = c;
        c.func();

        C c2 = cast(C)i;
        c2.func();

        c.func();
        assert(c.i == 4);
    }
}
