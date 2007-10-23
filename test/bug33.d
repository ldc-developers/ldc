module bug33;

extern(C) int memcmp(void*,void*,size_t);

private int string_cmp(char[] s1, char[] s2)
{
    auto len = s1.length;
    if (s2.length < len)
        len = s2.length;
    int result = memcmp(s1.ptr, s2.ptr, len);
    if (result == 0)
        result = cast(int)(cast(ptrdiff_t)s1.length - cast(ptrdiff_t)s2.length);
    return result;
}

struct S
{
    char[] toString()
    {
        return "S";
    }
}

int func()
{
    S a,b;
    return string_cmp(a.toString(),b.toString());
}

void main()
{
    assert(func() == 0);
}
