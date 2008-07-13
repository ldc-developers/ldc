module classes5;
extern(C) int printf(char*, ...);

struct S
{
    long l;
}

class C
{
    C c;
    S s;
}

void main()
{
    C c = new C;
    long* lp = void;
    {c.s.l = 64;}
    {assert(c.s.l == 64);}
    {lp = &c.s.l;}
    {assert(*lp == 64);}
    printf("classes5 success\n");
}
