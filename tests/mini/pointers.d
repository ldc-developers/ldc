module pointers;

struct S
{
    long l;
}

void main()
{
    int j = 42;
    int* p = &j;

    auto t = *p;
    *p ^= t;

    *p = ~t;

    S s;
    S* sp = &s;
    *sp = s;
    s = *sp;
}
