module complex1;

void main()
{
    cfloat cf1 = 3f + 0i;
    cfloat cf2 = 4f + 1i;
    cfloat cf3 = func();
    auto c1 = cf1 + cf2;
    auto c2 = cf2 - cf3;
    {
    auto c3 = cf1 * cf3;
    {
    auto c4 = cf2 / cf3;
    }
    }
}

cfloat func()
{
    return 3f + 1i;
}
