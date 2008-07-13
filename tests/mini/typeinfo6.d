module typeinfo6;

void main()
{
    auto ti = typeid(int[]);
    assert(ti.toString() == "int[]");
    assert(ti.next() is typeid(int));
}
