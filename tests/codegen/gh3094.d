// RUN: %ldc -run %s

void foo(void delegate() sink)
{
    return (0, sink());
}

void main()
{
    bool called;
    foo(() { called = true; });
    assert(called);
}
