// older LLVM versions fail the test for longer vectors
// REQUIRES: atleast_llvm700

// RUN: %ldc -run %s

void test(int length)()
{
    __vector(byte[length]) a = 123, b = a;
    assert(a == b && !(a != b));
    assert(a is b && !(a !is b));
    b[0] = 0;
    assert(a != b && !(a == b));
    assert(a !is b && !(a is b));
}

void main()
{
    static foreach (length; [16, 32, 64, 128, 256])
        test!length();
}
