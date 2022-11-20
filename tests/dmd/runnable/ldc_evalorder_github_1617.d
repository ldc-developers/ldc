int add8ret3(T)(ref T s)
{
    s += 8;
    return 3;
}

int mul11ret3(T)(ref T s)
{
    s *= 11;
    return 3;
}

void test_add()
{
    int val;
    val = 1;
    val += add8ret3(val);
    assert(val == (1 + 8 + 3));

    val = 1;
    val = val + add8ret3(val);
    assert(val == (1 + 3));

    val = 2;
    (val += 7) += mul11ret3(val);
    assert(val == (((2+7)*11) + 3));
}

void test_min()
{
    int val;
    val = 1;
    val -= add8ret3(val);
    assert(val == (1 + 8 - 3));

    val = 1;
    val = val - add8ret3(val);
    assert(val == (1 - 3));

    val = 2;
    (val -= 7) -= mul11ret3(val);
    assert(val == (((2-7)*11) - 3));
}

void test_mul()
{
    int val;
    val = 7;
    val *= add8ret3(val);
    assert(val == ((7 + 8) * 3));

    val = 7;
    val = val * add8ret3(val);
    assert(val == (7 * 3));

    val = 2;
    (val *= 7) *= add8ret3(val);
    assert(val == (((2*7)+8) * 3));
}

void test_xor()
{
    int val;
    val = 1;
    val ^= add8ret3(val);
    assert(val == ((1 + 8) ^ 3));

    val = 1;
    val = val ^ add8ret3(val);
    assert(val == (1 ^ 3));

    val = 2;
    (val ^= 7) ^= add8ret3(val);
    assert(val == (((2^7)+8) ^ 3));
}

void test_addptr()
{
    int* val;
    val = cast(int*)4;
    val += add8ret3(val);
    assert(val == ((cast(int*)4) + 8 + 3));

    val = cast(int*)4;
    val = val + add8ret3(val);
    assert(val == ((cast(int*)4) + 3));

    val = cast(int*)16;
    (val += 7) += add8ret3(val);
    assert(val == ((cast(int*)16) + 7 + 8 + 3));
}

void test_lhsCast()
{
    byte val = 1;
    // lhs type `byte`, rhs type `int` =>
    // rewritten to `cast(int)(cast(int)val += 10) -= mul11ret3(val)`
    (val += 10) -= mul11ret3(val);
    assert(val == ((1 + 10) * 11 - 3));
}

void main()
{
    test_add();
    test_min();
    test_mul();
    test_xor();
    test_addptr();
    test_lhsCast();
}
