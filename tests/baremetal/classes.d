// Use classes with virtual functions.

// RUN: %ldc %baremetal_args -run %s

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
    if (obj.isB())
        return 1;

    obj = b;
    if (!obj.isB())
        return 2;

    return 0;
}
