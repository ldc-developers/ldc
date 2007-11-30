module nested5;

void main()
{
    int i = 42;

    printf("Hello world %d\n", i++);

    class C
    {
        void func()
        {
            printf("Hello world %d\n", i++);
        }
    }

    scope c = new C;
    c.func();
}
