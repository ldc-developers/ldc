module memory1;

extern(C) int printf(char*,...);

void main()
{
    auto a = new int[16];
    {printf("array.length = %u\n", a.length);}
    {a.length = a.length + 1;}
    {printf("array.length = %u\n", a.length);}
    {assert(a.length == 17);}
}
