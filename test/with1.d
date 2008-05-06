module with1;
struct S
{
    int i;
    float f;
}
void main()
{
    S s;
    with(s)
    {
        i = 0;
        f = 3.5;
    }
    assert(s.i == 0);
    assert(s.f == 3.5);
}
