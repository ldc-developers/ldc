// Tests that a small non-POD isn't implicitly blit when being passed to/
// returned from a called function.

// RUN: %ldc -run %s

struct S
{
    S* self;
    this(this) { self = &this; }
    ~this() { assert(self == &this); }
}

S foo(S param)
{
    return param; // copy-construct return value
    // destruct param
}

void main()
{
    S s;
    s.self = &s;

    S r = foo(s); // copy-construct tmp arg from s
    // destruct r and s
}
