module nested12;

extern(C) int printf(char*, ...);

void main()
{
    func();
}

void func()
{
    void a(int i)
    {
        printf("%d\n", i);
    }

    void b()
    {
        a(42);
    }

    b();
}
