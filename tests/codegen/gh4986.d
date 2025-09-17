// RUN: %ldc %s -c

struct Test
{
    this(ref inout typeof(this) rhs) inout pure {}
    const(char)[] toString() const pure => null;
    alias toString this;
}

const(char)[] test(ref Test t)
{
    return t;
}
