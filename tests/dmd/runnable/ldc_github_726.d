
void issue726_1()
{
    struct Buggy
    {
        align(1):
        uint a = 0x0a0a0a0a;
        ulong b = 0x0b0b0b0b0b0b0b0b;
    }

    Buggy packed;
    ulong raw = *cast(ulong*)(cast(ubyte*)&packed + packed.b.offsetof);
    assert(packed.b == raw);
}

void issue726_2()
{
    class Buggy
    {
        align(1):
        uint a = 0x0a0a0a0a;
        ulong b = 0x0b0b0b0b0b0b0b0b;
    }

    auto packed = new Buggy;
    ulong raw = *cast(ulong*)(cast(ubyte*)packed + packed.b.offsetof);
    assert(packed.b == raw);
}

void issue726_3()
{
    class Buggy
    {
        align(1):
        uint a = 0x0a0a0a0a;
        ulong b = 0x0b0b0b0b0b0b0b0b;
    }

    class Derived : Buggy
    {
    }

    auto packed = new Derived;
    ulong raw = *cast(ulong*)(cast(ubyte*)packed + packed.b.offsetof);
    assert(packed.b == raw);
}

void issue726_4()
{
    struct Buggy
    {
        align(1):
        uint a = 0x0a0a0a0a;
        ulong b = 0x0b0b0b0b0b0b0b0b;

        align(8):
        ulong c = 0x0c0c0c0c0c0c0c0c;
    }

    Buggy packed;

    ulong raw = *cast(ulong*)(cast(ubyte*)&packed + packed.b.offsetof);
    assert(packed.b == raw);

    raw = *cast(ulong*)(cast(ubyte*)&packed + packed.c.offsetof);
    assert(packed.c == raw);
}

void main()
{
    issue726_1();
    issue726_2();
    issue726_3();
    issue726_4();
}