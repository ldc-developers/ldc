class A
{
    int i;
    void f()
    {
        printf("A.f\n");
    }
}

class B : A
{
    long l;
    void f()
    {
        printf("B.f\n");
    }
}

void main()
{
    A a = new B;
    a.f();
}
