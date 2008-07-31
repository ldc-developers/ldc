module union7;

struct Union
{
    union {
        double g;
        struct {
            short s1,s2,s3,s4;
        }
    }
    union {
        float f;
        long l;
    }
}

Union a = { f:4f };
Union b = { 3.0, f:2 };
Union c = { l:42, g:2.0 };
Union d = { s2:3 };
Union e = { s1:3, s4:4, l:5 };

void main()
{
    assert(a.f == 4f);
    assert(a.g !<>= 0.0);
    assert((a.l>>>32) == 0);

    assert(b.g == 3.0);
    assert(b.f == 2f);

    assert(c.l == 42);
    assert(c.g == 2.0);

    assert(d.s1 == 0);
    assert(d.s2 == 3);
    assert(d.s3 == 0);
    assert(d.s4 == 0);
    {assert(d.f !<>= 0f);}
    {}
    assert(e.s1 == 3);
    assert(e.s2 == 0);
    assert(e.s3 == 0);
    {assert(e.s4 == 4);}
    {}
    assert(e.l == 5);
}
