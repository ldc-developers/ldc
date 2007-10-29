module bug39;

struct S
{
    long l;
}

void main()
{
    S s;
    s.l = 23;
    void* p = &s;
}
