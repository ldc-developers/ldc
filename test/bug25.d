module bug25;

void main()
{
    int i = 2;
    delegate {
        i = i*i;
        i += i*i;
    }();
    printf("%d\n", i);
    assert(i == 20);
}
