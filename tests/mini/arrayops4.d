void main()
{
    auto a = new float[1024];
    auto b = new float[1024];
    auto c = new float[1024];

    for (auto i=0; i<1024; i++)
    {
        a[i] = i;
        b[i] = i*2;
        c[i] = i*4;
    }

    a[] = b[] + c[] / 2;

    foreach(i,v; a)
    {
        assert(eq(v, b[i] + c[i] / 2));
    }
}

float abs(float x)
{
    return x<0?-x:x;
}
bool eq(float a, float b)
{
    return abs(a-b) <= float.epsilon;
}
