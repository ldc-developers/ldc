// RUN: %ldc -run %s

int bar() { return 0; }

void foo(void delegate() sink)
{
    return (bar(), sink());
}

void main()
{
    bool called;
    foo(() { called = true; });
    assert(called);
}
