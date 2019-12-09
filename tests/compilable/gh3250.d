// RUN: %ldc -c %s

struct Q
{
    auto func(uint[3] a, uint[3] b, uint c)
    {
        return this;
    }
}
