// RUN: %ldc -c %s

void foo()
{
    extern(C) void DoubleArrayToAnyArray(void* arg0)
    {
        auto dg = () => { auto r = arg0; };
    }
    auto local = 123;
    auto arg = () { return local; }();
    DoubleArrayToAnyArray(null);
}
