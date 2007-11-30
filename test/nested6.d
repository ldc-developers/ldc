module nested6;

void main()
{
    int i = 42;

    printf("Hello world %d\n", i++);

    class C
    {
        void func()
        {
            printf("Hello world %d\n", i++);

            class C2
            {
                void func2()
                {
                    printf("Hello world %d\n", i++);
                }
            }

            {
                scope c2 = new C2;
                c2.func2();
            }
        }
    }

    scope c = new C;
    c.func();
}
