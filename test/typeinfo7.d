module typeinfo7;

int func(long)
{
    return 0;
}

void main()
{
    auto ti = typeid(typeof(func));
    auto s = ti.toString;
    printf("%.*s\n", s.length, s.ptr);
    assert(s == "int()");
}
