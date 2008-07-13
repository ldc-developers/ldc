module bug24;
extern(C) int printf(char*, ...);

struct S
{
    long l;
    float f;
}

void main()
{
    S s = S(3L,2f);
    delegate {
        S t = S(4L, 1f);
        delegate {
            s.l += t.l;
            s.f += t.f;
        }();
    }();
    printf("%lu %f\n", s.l, s.f);
    assert(s.l == 7 && s.f == 3);
}
