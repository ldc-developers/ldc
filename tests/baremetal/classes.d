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

// Test requires linking with C standard library
extern(C) void exit(int status);

extern(C) int main()
{
    A obj = a;
    if (obj.isB())
        exit(1);

    obj = b;
    if (!obj.isB())
        exit(1);

    return 0;
}
