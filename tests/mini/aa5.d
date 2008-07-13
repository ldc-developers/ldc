module aa5;

void main()
{
    int[int] aa;
    aa[42] = 1;
    int i = aa[42];
    assert(i == 1);
}
