module bug50;
extern(C) int printf(char*, ...);

pragma(LLVM_internal, "notypeinfo")
struct S
{
    int i;
    float f;
    long l;

    void print()
    {
        printf("%d %f %lx\n", i, f, l);
    }
}

void main()
{
    S s;
    s.print();
    s = S(1,2,3);
    s.print();

    S[] arr;
    {arr ~= s;}
    {arr[0].print();}
    {arr ~= S(1,2,3);}
    {arr[1].print();}
}
