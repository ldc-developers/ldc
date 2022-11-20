__gshared bool baseMethodCalled = false;

struct S { long a, b; }

class A
{
    S foo()
    {
        baseMethodCalled = true;
        return S();
    }
}

class B : A
{
    override S foo()
    {
        auto r = super.foo();
        return r;
    }
}

void main()
{
    auto b = new B;
    b.foo();
    assert(baseMethodCalled);
}
