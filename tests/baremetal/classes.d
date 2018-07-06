// Use classes with virtual functions.

// -betterC for C assert.
// RUN: %ldc -betterC %baremetal_args -run %s

class A
{
    int x;
    bool isB() { return false; }
}

class B : A
{
    override bool isB() { return true; }
}

__gshared A a = new A();
__gshared B b = new B();

extern(C) int main()
{
    A obj = a;
    assert(!obj.isB());

    obj = b;
    assert(obj.isB());

    return 0;
}
