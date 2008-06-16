module classes11;
extern(C) int printf(char*, ...);

void main()
{
    static class C
    {
        void func()
        {
            printf("Hello world\n");
        }
    }

    scope c = new C;
    c.func();
}
