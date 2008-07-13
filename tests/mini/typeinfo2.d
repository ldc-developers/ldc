module typeinfo2;

void main()
{
    auto ti = typeid(float);
    float f = 2.5;
    hash_t fh = ti.getHash(&f);
    assert(ti.next is null);
    float g = 4.0;
    ti.swap(&f,&g);
    assert(f == 4.0 && g == 2.5);
    assert(fh == *cast(uint*)(&g));
    assert(!ti.flags);
}
