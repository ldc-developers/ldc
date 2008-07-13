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
            printf("Hello world %d\n", i++);
            //i++;
        }
    }

    scope c = new C;
    c.func();
    assert(i == 44);
}
