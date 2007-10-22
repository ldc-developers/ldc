module bug23;
void main()
{
    int i;
    delegate {
        i++;
        delegate {
            i++;
        }();
    }();
    printf("%d\n", i);
    assert(i == 2);
}
