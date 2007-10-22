module bug20;

void func(void delegate() dg)
{
    dg();
}

void main()
{
    int i = 42;
    void delegate() dg = {
        i++;
    };
    printf("i = %d\n",i);
    func(dg);
    printf("i = %d\n",i);
    assert(i == 43);
}
