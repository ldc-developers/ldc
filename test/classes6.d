module classes6;

class C
{
    void f()
    {
        printf("hello world\n");
    }
}

void main()
{
    scope c = new C;
    c.f();
}
