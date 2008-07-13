module nested6;
extern(C) int printf(char*, ...);

void main()
{
    int i = 42;

    printf("main() %d\n", i++);

    class C
    {
        void func()
        {
            printf("C.func() %d\n", i++);

            class C2
            {
                void func2()
                {
                    printf("C2.func2() %d\n", i++);
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
