module classinfo4;

class C
{
}

class D
{
    this()
    {
    }
}

void main()
{
    assert(C.classinfo.defaultConstructor is null);
    assert(D.classinfo.defaultConstructor !is null);
}
