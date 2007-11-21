module typeinfo13;

void main()
{
    float[long] aa;
    auto ti = typeid(typeof(aa));
    assert(ti.toString() == "float[long]");
    assert(ti.next() is typeid(float));
    assert(ti.tsize() == size_t.sizeof);
    auto aati = cast(TypeInfo_AssociativeArray)ti;
    assert(aati.value is typeid(float));
    assert(aati.key is typeid(long));
}
