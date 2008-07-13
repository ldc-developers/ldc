module bug25;
extern(C) int printf(char*, ...);

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
