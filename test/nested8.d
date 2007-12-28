module nested8;

void main()
{
    int i = 1;
    void func()
    {
        printf("func()\n");
        i++;
        void func2()
        {
            printf(" func2()\n");
            int j = i + 1;
            void func3()
            {
                printf("  func3()\n");
                j++;
                printf("  done = %d\n", j);
            }
            func3();
            i = j;
            printf(" done = %d\n", j);
        }
        func2();
        printf("done\n");
    }
    func();
    printf("i == %d\n", i);
    assert(i == 4);
}
