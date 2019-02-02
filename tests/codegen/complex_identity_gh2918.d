// RUN: %ldc -run %s

void main()
{
    creal f1 = +0.0 + 0.0i;
    creal f2 = +0.0 - 0.0i;
    creal f3 = -0.0 + 0.0i;
    creal f4 = +0.0 + 0.0i;
    assert(f1 !is f2);
    assert(f1 !is f3);
    assert(f2 !is f3);
    assert(f1 is f4);
    assert(!(f1 is f2));
    assert(!(f1 is f3));
    assert(!(f2 is f3));
    assert(!(f1 !is f4));
}
