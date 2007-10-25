module arrays8;

void main()
{
    char[] a = "hello ";
    printf("  \"%s\".length = %u\n", a.ptr, a.length);
    char[] b = "world";
    printf("  \"%s\".length = %u\n", b.ptr, b.length);
    char[] c = a ~ b;
    printf("After 'a ~ b':\n");
    printf("  \"%.*s\".length = %u\n", a.length, a.ptr, a.length);
    printf("  \"%.*s\".length = %u\n", b.length, b.ptr, b.length);
    printf("  \"%.*s\".length = %u\n", c.length, c.ptr, c.length);
    assert(c.length == a.length + b.length);
    assert(c !is a);
    assert(c !is b);
}
