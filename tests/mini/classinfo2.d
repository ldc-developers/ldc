module classinfo2;

class C
{
}

class D : C
{
}

void main()
{
    D d;
    d = new D;
    ClassInfo ci = d.classinfo;
    assert(ci is D.classinfo);
}
