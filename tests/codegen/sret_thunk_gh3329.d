// RUN: %ldc -run %s

extern(C) int printf(const(char)* format, ...);

struct NoPOD
{
    size_t x;
    ~this() {}
}

interface I
{
    NoPOD doIt(size_t arg);
}

__gshared C c;

class C : I
{
    this()
    {
        c = this;
        printf("c: %p\n", c);
    }

    NoPOD doIt(size_t arg)
    {
        printf("doIt this: %p; arg: %p\n", this, arg);
        assert(this == c);
        assert(arg == 0x2A);
        return NoPOD(arg << 4);
    }
}

void main()
{
    I i = new C;
    printf("i: %p\n", i);
    NoPOD r = i.doIt(0x2A);
    printf("&r: %p\n", &r);
    assert(r.x == 0x2A0);
}
