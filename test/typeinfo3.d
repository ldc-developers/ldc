module typeinfo3;

typedef int int_t;

void main()
{
    int_t i;
    auto ti = typeid(typeof(i));
    printf("%s\n",ti.toString.ptr);
    assert(ti.toString() == "typeinfo3.int_t");
    assert(ti.next !is null);
    assert(ti.next.toString() == "int");
}
