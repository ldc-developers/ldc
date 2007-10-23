module bug28;

void main()
{
    char[] a = "hello";
    char[] b = "hello";
    char[] c = "world";
    char[] d = "somethingelse";
    assert(a == a);
    assert(a == b);
    assert(a != c);
    assert(b != c);
    assert(a != d);
    assert(b != d);
    assert(c != d);
    assert(d == d);
}
