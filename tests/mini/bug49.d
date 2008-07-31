module bug49;

struct S
{
    int i;
    long l;
}

void main()
{
    S s;
    s.i = 0x__FFFF_FF00;
    s.l = 0xFF00FF_FF00;
    s.i &= s.l;
}
