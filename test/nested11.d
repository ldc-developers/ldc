module nested11;

void main()
{
    int i;

    void f()
    {
        i++;

        void g()
        {
            i++;

            void h()
            {
                printf("i = %d\n", i);
            }

            h();
        }

        g();
    }

    f();
    assert(i == 2);

    void foo()
    {
        i = 42;
    }

    void bar()
    {
        foo();
    }

    bar();
    printf("i = %d\n", i);
    assert(i == 42);
}
