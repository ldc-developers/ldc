module interface6;

interface I
{
    void Ifunc();
}

interface J
{
    void Jfunc();
}

class C : I,J
{
    int i;
    int j;
    void Ifunc()
    {
        i++;
    }
    void Jfunc()
    {
        j++;
    }
}

void main()
{
    C c = new C;
    c.Ifunc();
    c.Jfunc();
    I i = c;
    i.Ifunc();
    J j = c;
    j.Jfunc();
    C c2 = cast(C)i;
    c2.Ifunc();
    c2.Jfunc();
    C c3 = cast(C)j;
    c3.Ifunc();
    c3.Jfunc();
    assert(c.i == 4);
    assert(c.j == 4);
}
