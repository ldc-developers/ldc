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

struct Result(Type) {
    Type get() {  assert(0); }

    alias get this;

    this(ref Result) {}
}

float dotProduct() {
    Result!float got;
    return got;
}
