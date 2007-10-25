module typeinfo5;

void main()
{
    enum E : uint {
        A,B,C
    }
    auto ti = typeid(E);
    assert(ti.next() is typeid(uint));
}
