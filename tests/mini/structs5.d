module structs5;

void main()
{
    {S s = S();}
    {T t = T(1);}
    {U u = U();}
}

struct S
{
}

struct T
{
    int i;
}

struct U
{
    S s;
    long l;
}
