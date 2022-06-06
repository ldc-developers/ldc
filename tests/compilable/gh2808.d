// RUN: %ldc -c %s

void gh2808()
{
    extern(C) void DoubleArrayToAnyArray(void* arg0)
    {
        auto dg = () => { auto r = arg0; };
    }
    auto local = 123;
    auto arg = () { return local; }();
    DoubleArrayToAnyArray(null);
}

void gh3234()
{
    int i;
    void nested() { ++i; }

    extern (C++) class Visitor
    {
        void visit() { nested(); }
    }
}
