module condexp1;

void main()
{
    char[] a = "hello";
    char[] b = "world";
    int i = 42;
    {
    char[] c = i > 50 ? b : a;
    assert(c is a);
    }
}
