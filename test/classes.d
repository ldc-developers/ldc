class C
{
    int i;
    void p()
    {
        printf("%d\n", i);
    }
}

void main()
{
    printf("should print 4\n");
    C c = new C;
    c.i = 4;
    c.p();
    //delete c;
}
