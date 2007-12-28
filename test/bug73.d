int find(char[] s, dchar c)
{
    // c is a universal character
    foreach (int i, dchar c2; s)
    {
    if (c == c2)
        return i;
    }
    return -1;
}

void main()
{
    char[] hello = "hello world";
    int i = find(hello, 'w');
    assert(i == 6);
    i = find(hello, 'z');
    assert(i == -1);
}
