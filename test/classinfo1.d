module classinfo1;

class NoPtrs
{
}

class HasPtrs
{
    void* p;
}

void main()
{
    {
    ClassInfo ci = NoPtrs.classinfo;
    char[] name = ci.name;
    printf("%.*s\n", name.length, name.ptr);
    assert(ci.name == "classinfo1.NoPtrs");
    assert(ci.flags & 2);
    }
    {
    ClassInfo ci = HasPtrs.classinfo;
    char[] name = ci.name;
    printf("%.*s\n", name.length, name.ptr);
    assert(ci.name == "classinfo1.HasPtrs");
    assert(!(ci.flags & 2));
    }
}
