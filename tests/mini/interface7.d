module interface7;

extern(C) int printf(char*,...);

interface I
{
}

class C : I
{
}

void main()
{
    I i = new C;
    ClassInfo ci = i.classinfo;
    char[] name = ci.name;
    printf("ci.name = %.*s\n", name.length, name.ptr);
    ClassInfo cI = I.classinfo;
    name = cI.name;
    printf("cI.name = %.*s\n", name.length, name.ptr);
    assert(ci is cI);
}
