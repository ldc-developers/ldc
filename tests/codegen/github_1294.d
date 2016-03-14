// RUN: %ldc -c -of=%t %s

__gshared bool a;

struct BigInt
{
    this(T)(T)
    {
    }

    struct Payload
    {
        ~this()
        {
        }
    }

    Payload data;
}

BigInt randomPrime()
{
    return a ? BigInt(123) : BigInt(456);
}
