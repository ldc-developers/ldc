module nested6a;
extern(C) int printf(char*, ...);

void main()
{
    int i = 42;

    printf("main() %d\n", i++);

    class C
    {
        int j;
        void func()
        {
            int k;
            printf("C.func() %d\n", i++);

            class C2
            {
                int l;
                void func2()
                {
                    printf("in C2.func2()\n");
                    printf("C2.func2() %d\n", i++);
                }
                int m;
            }

            {
                printf("new C2\n");
                auto c2 = new C2;
                printf("C2.func2()\n");
                c2.func2();
            }
        int n;
        }
    int o;
    }

    auto c = new C;
    c.func();
}
