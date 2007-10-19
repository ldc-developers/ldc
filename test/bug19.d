module bug19;

void main()
{
    auto dg = (int i) { return i*2; };
    assert(dg(2) == 4);
}
