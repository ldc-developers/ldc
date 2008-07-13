module nested7;
extern(C) int printf(char*, ...);

void main()
{
    int i;
    i = 52;
    printf("i = %d\n", i);

    void func()
    {
        i++;

        void func2()
        {
            i++;

            void func3()
            {
                i++;
            }

            func3();
        }

        func2();
    }

    func();

    printf("i = %d\n", i);
    assert(i == 55);
}
