module typeinfo10;

struct S
{
    long l;
    float f;
    void* vp;

    hash_t toHash()
    {
        return l + cast(size_t)f;
    }

    int opEquals(S s)
    {
        return (s.l == l) && (s.f == f);
    }

    int opCmp(S a)
    {
        if (l == a.l) {
            return (f < a.f) ? -1 : (f > a.f) ? 1 : 0;
        }
        return (l < a.l) ? -1 : 1;
    }

    char[] toString()
    {
        return "S instance";
    }
}

void main()
{
    S s=S(-1, 0);
    S t=S(-1, 1);
    S u=S(11,-1);
    S v=S(12,13);

    {
        assert(s == s);
        assert(s != t);
        assert(s != v);
        assert(s < t);
        assert(u > s);
        assert(v > u);
    }

    {
        auto ti = typeid(S);
        assert(ti.getHash(&s) == s.toHash());
        assert(ti.equals(&s,&s));
        assert(!ti.equals(&s,&t));
        assert(!ti.equals(&s,&v));
        assert(ti.compare(&s,&s) == 0);
        assert(ti.compare(&s,&t) < 0);
        assert(ti.compare(&u,&s) > 0);
        assert(ti.compare(&v,&u) > 0);
        {
            auto tis = cast(TypeInfo_Struct)ti;
            char[] delegate() structToString;
            structToString.ptr = &s;
            structToString.funcptr = tis.xtoString;
            assert(structToString() == s.toString());
        }
    }
}
