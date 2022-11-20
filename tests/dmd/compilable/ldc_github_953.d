// Makes sure the compiler doesn't attempt to destruct a temporary inside an
// assert() expression if assertions are disabled.
// REQUIRED_ARGS: -release

struct S
{
    int v;
    ~this() {}
}

S foo() { return S(3); }

void main()
{
    assert(foo().v == 3);
}
