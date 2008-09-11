module complex5;

void main()
{
    cfloat c = 3+2i;
    foo(c);
}

void foo(cfloat c)
{
    assert(c.re > 2.9999  && c.re < 3.0001);
    assert(c.im > 1.9999i && c.im < 2.0001);
}
