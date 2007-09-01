void main()
{
    printf("Pointer arithmetic test\n");
    int* p;
    printf("0x%x\n", p);
    assert(p++ is null);
    assert(cast(size_t)p == 4);
    printf("0x%x\n", p);
    p--;
    assert(p is null);
    printf("0x%x\n", p);
    int d = 4;
    p+=d;
    printf("0x%x\n", p);
    assert(cast(size_t)p == 16);
    d = 2;
    p+=d;
    printf("0x%x\n", p);
    assert(cast(size_t)p == 0x18);
    d = 6;
    p-=d;
    printf("0x%x\n", p);
    assert(p is null);
    printf("  SUCCESS\n");
}

void fill_byte_array(ubyte* a, size_t n, ubyte v)
{
    auto p = a;
    auto end = a+n;
    while (p !is end)
        *p++ = v;
}
