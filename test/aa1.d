module aa1;

void main()
{
    int[int] aai;
    assert(aai is null);
    aai[0] = 52;
    assert(aai !is null);
    int i = aai[0];
    assert(i == 52);
    aai[32] = 123;
    int j = aai[32];
    assert(i == 52);
    assert(j == 123);
}
