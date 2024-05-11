// RUN: %ldc -run %s

void main()
{
    int x;

    assert(!(++x < x++)); // CmpExp: !(1 < 1)
    assert(x == 2);

    assert(!(++x > x++)); // CmpExp: !(3 > 3)
    assert(x == 4);

    assert(++x == x++); // EqualExp: 5 == 5
    assert(x == 6);

    assert(++x is x++); // IdentityExp: 7 is 7
    assert(x == 8);

    assert(x++ !is x); // IdentityExp: 8 !is 9
    assert(x == 9);
}
