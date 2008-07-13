module structs3;

struct S
{
    char c;
    float f;
}

struct T
{
    S s;
    long l;
}

void main()
{
    T t;
    float f = void;
    float* fp = void;
    {f = t.s.f;}
    {t.s.f = 0.0;}
    {fp = &t.s.f;}
    {*fp = 1.0;}
    {assert(t.s.f == 1.0);}
    {assert(*(&t.s.f) == 1.0);}
    {t.s.c = 'a';}
    {assert(t.s.c == 'a');}
    {t.l = 64;}
    {assert(t.l == 64);}
}
