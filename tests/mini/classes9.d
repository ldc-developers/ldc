module classes9;

class C
{
}

class D : C
{
}

class E
{
}

class F : E
{
}

void main()
{
    {
        D d = new D;
        {
            C c = d;
            assert(c !is null);
            D d2 = cast(D)c;
            assert(d2 !is null);
            E e = cast(E)d;
            assert(e is null);
            F f = cast(F)d;
            assert(f is null);
        }
    }
    {
        F f = new F;
        {
            E e = f;
            assert(e !is null);
            F f2 = cast(F)e;
            assert(f2 !is null);
            C c = cast(C)f;
            assert(c is null);
            D d2 = cast(D)f;
            assert(d2 is null);
        }
    }
}
