// RUN: %ldc -run %s

void test(int length)()
{
    alias V = __vector(byte[length]);

    V a = 123, b = a;

    V eqMask = -1;
    V neqMask = 0;

    assert(a is b && !(a !is b));
    assert((a == b) is eqMask);
    assert((a != b) is neqMask);

    b[0] = 0;
    eqMask[0] = 0;
    neqMask[0] = -1;

    assert(a !is b && !(a is b));
    assert((a == b) is eqMask);
    assert((a != b) is neqMask);
}

void main()
{
    static foreach (length; [16, 32, 64, 128, 256])
        test!length();
}
