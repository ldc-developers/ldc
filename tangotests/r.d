extern(C) int printf(char*, ...);

class C
{
    void dump()
    {
        printf("C dumped\n");
    }
}

void heap()
{
    auto c = new C;
    c.dump();
}

void stack()
{
    scope c = new C;
    c.dump();
}

void main()
{
    heap();
    stack();
}
