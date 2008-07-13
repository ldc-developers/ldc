module foreach7;

void main()
{
    int[4] a = [1,2,3,4];
    int i;
    foreach(v; a[0..3])
    {
        i += v;
    }
    assert(i == 6);
}
