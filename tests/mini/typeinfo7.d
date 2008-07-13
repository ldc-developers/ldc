module typeinfo7;
extern(C) int printf(char*, ...);
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
