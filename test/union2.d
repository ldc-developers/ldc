module union2;

pragma(LLVM_internal, "notypeinfo")
union U
{
    float f;
    long l;
}

U u;

void main()
{
    assert(u.f !<>= 0);
    uint* p = 1 + cast(uint*)&u.l;
    assert(*p == 0);
}
