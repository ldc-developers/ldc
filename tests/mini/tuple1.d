module tuple1;

template Tuple(T...) {
    alias T Tuple;
}

struct S
{
    int i;
    long l;
}

void main()
{
    S s = S(Tuple!(1,2L));
    assert(s.i == 1);
    assert(s.l == 2);
}
