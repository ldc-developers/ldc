module tangotests.loops1;

void main()
{
    size_t n;
    int x;
    ushort foo;
    for (n=0; n<8; n++,x++)
    {
        foo >>= 1;
    }
}
