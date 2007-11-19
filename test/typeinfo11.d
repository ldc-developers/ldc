module typeinfo11;

void main()
{
    int[4] a;
    TypeInfo ti;
    ti = typeid(typeof(a));
    assert(ti.next() is typeid(int));
    assert(ti.tsize() == 16);
}
