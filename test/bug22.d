module bug22;

void main()
{
    int i;
    delegate {
        i = 42;
    }();
    printf("%d\n", i);
    assert(i == 42);
}
