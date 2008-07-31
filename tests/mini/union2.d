module union2;

union U
{
    float f;
    long l;
}

U u;

void main()
{
    assert(u.f !<>= 0);
    {
        uint* p = 1 + cast(uint*)&u;
        {assert(*p == 0);}
    }
}
