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
}