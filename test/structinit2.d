module structinit2;

struct Imp
{
    int i;
    long l;
    float f;
}

void main()
{
    Imp i;
    assert(i.i == 0);
    assert(i.l == 0L);
    assert(i.f !<>= 0.0f);
}
