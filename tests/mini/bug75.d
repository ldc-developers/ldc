module bug75;

void func(void delegate() dg)
{
}

void main()
{
    void nested() {
    }
    //func(&nested);
    void delegate() dg = &nested;
}
