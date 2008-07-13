module test.structs8;

struct S
{
    int a,b;
}

void main()
{
    S a = S(1,2);
    S b = S(2,3);
    S c = S(3,4);
    S d = S(2,3);

    assert(a == a);
    assert(a != b);
    assert(a != c);
    assert(a != d);

    assert(b != a);
    assert(b == b);
    assert(b != c);
    assert(b == d);

    assert(c != a);
    assert(c != b);
    assert(c == c);
    assert(c != d);

    assert(d != a);
    assert(d == b);
    assert(d != c);
    assert(d == d);

    assert(a is a);
    assert(a !is b);
    assert(a !is c);
    assert(a !is d);

    assert(b !is a);
    assert(b is b);
    assert(b !is c);
    assert(b is d);

    assert(c !is a);
    assert(c !is b);
    assert(c is c);
    assert(c !is d);

    assert(d !is a);
    assert(d is b);
    assert(d !is c);
    assert(d is d);
}
