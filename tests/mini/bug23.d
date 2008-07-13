module bug23;
extern(C) int printf(char*, ...);

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
