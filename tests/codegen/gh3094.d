// RUN: %ldc -run %s

void foo(void delegate() sink)
{
    return sink();
}

void main()
{
    bool called;
    foo(() { called = true; });
    assert(called);
}
