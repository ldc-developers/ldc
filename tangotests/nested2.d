module tangotests.nested2;

extern(C) int printf(char*, ...);

void main()
{
    int var = 2;

    void exec(void delegate() dg)
    {
        printf("var = %d\n", var);
        dg();
    }

    void foo()
    {
        printf("var = %d\n", var);
        assert(var == 5);
    }

    void bar()
    {
        printf("var = %d\n", var);
        var += 3;
        exec(&foo);
    }

    printf("var = %d\n", var);
    exec(&bar);

    return 0;
}
