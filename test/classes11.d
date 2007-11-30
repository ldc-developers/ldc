module classes11;

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
