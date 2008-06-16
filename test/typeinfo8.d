module typeinfo8;
extern(C) int printf(char*, ...);
struct S
{
    void func()
    {
    }
}

void main()
{
    S a;
    auto ti = typeid(typeof(&a.func));
    auto s = ti.toString;
    printf("%.*s\n", s.length, s.ptr);
    assert(s == "void delegate()");
}
