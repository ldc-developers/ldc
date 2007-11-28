module bug78;

void main()
{
    typedef int int_t = 42;
    int_t i;
    assert(i == int_t.init);
}