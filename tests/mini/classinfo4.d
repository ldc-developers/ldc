module classinfo4;

class C
{
}

class D : C
{
    this()
    {
    }
    ~this()
    {
    }
}

template T()
{
    ~this()
    {
    }
}

class E : D
{
    this()
    {
    }
    ~this()
    {
    }
    mixin T;
}

void main()
{
    assert(C.classinfo.defaultConstructor is null);
    assert(C.classinfo.destructor is null);
    assert(D.classinfo.defaultConstructor !is null);
    assert(D.classinfo.destructor !is null);
    assert(E.classinfo.defaultConstructor !is null);
    assert(E.classinfo.destructor !is null);
}
