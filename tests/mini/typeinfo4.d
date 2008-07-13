module typeinfo4;

void main()
{
    auto ti = typeid(void*);
    assert(ti.toString() == "void*");
    assert(ti.tsize() == size_t.sizeof);
    void* a = null;
    void* b = a + 1;
    assert(ti.compare(&a,&b) < 0);
}
