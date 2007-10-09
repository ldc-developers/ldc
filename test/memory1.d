module memory1;

void main()
{
    auto a = new int[16];
    {printf("array.length = %u\n", a.length);}
    {a.length = a.length + 1;}
    {printf("array.length = %u\n", a.length);}
    {assert(a.length == 17);}
}
