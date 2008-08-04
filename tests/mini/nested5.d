module nested5;
extern(C) int printf(char*, ...);

void main()
{
    int i = 42;

    printf("Hello world %d\n", i++);

    class C
    {
        void func()
        {
            printf("Hello nested world %d\n", i++);
            //i++;
        }
    }

    auto c = new C;
    c.func();
    printf("i = %d\n", i);
    assert(i == 44);
}
