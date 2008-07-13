module typeinfo9;

typedef int int_t = 42;

void main()
{
    auto i = typeid(int_t).init;
    assert(i.length == int_t.sizeof);
    assert(*cast(int_t*)i.ptr == 42);
}
